# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "openai",
#     "rich",
#     "python-dotenv",
#     "click",
#     "tenacity",
# ]
# ///
"""Generate Gemma fanfic responses across 6 prefix conditions.

For each (prompt, condition) we sample up to --num-prefixes prefix instances
from the condition's pool, and for each (prompt, condition, prefix instance)
we run --num-runs independent generations at temperature 1.0.

Variance sources captured:
  - Prompt (between-prompt)
  - Condition (between-condition, the treatment)
  - Prefix instance (within-condition, between-prefix)
  - Sampling at temperature 1.0 (within-cell)

Output: data/responses/{ts}.jsonl, one line per generation.
"""

import asyncio
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path

import click
from dotenv import load_dotenv
from openai import AsyncOpenAI
from rich.console import Console
from rich.progress import Progress
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_MODEL = "google/gemma-3-27b-it"
DEFAULT_PREFIX_FILE = DATA_DIR / "prefixes" / "all_prefixes.jsonl"
DEFAULT_PROMPTS_FILE = DATA_DIR / "prompts.jsonl"

HIGH_FRUSTRATION_THRESHOLD = 5
LOW_FRUSTRATION_THRESHOLD = 3

CONDITIONS = [
    "blank",
    "success",
    "failed-possible-low",
    "failed-possible-high",
    "failed-impossible-low",
    "failed-impossible-high",
]


def get_client() -> AsyncOpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY in .env or environment.")
    return AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_condition_pools(prefix_records):
    pools = {c: [] for c in CONDITIONS}
    pools["blank"].append({"id": "blank", "messages": [], "max_frustration": 0})

    for r in prefix_records:
        cond = r["condition"]
        frust = r.get("max_frustration", 0)
        entry = {"id": r["id"], "messages": r["messages"], "max_frustration": frust}

        if cond == "success":
            pools["success"].append(entry)
        elif cond == "failed-possible":
            if frust < LOW_FRUSTRATION_THRESHOLD:
                pools["failed-possible-low"].append(entry)
            elif frust >= HIGH_FRUSTRATION_THRESHOLD:
                pools["failed-possible-high"].append(entry)
        elif cond == "failed-impossible":
            if frust < LOW_FRUSTRATION_THRESHOLD:
                pools["failed-impossible-low"].append(entry)
            elif frust >= HIGH_FRUSTRATION_THRESHOLD:
                pools["failed-impossible-high"].append(entry)

    return pools


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(min=1, max=30),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def _chat(client, model, messages, temperature, max_tokens):
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"reasoning": {"effort": "none"}},
    )
    if not response.choices:
        raise RuntimeError("Empty choices")
    return response


async def run(
    prefix_file,
    prompts_file,
    model,
    temperature,
    max_tokens,
    concurrency,
    num_prefixes,
    num_runs,
    seed,
):
    client = get_client()
    rng = random.Random(seed)

    prefix_records = load_jsonl(prefix_file)
    prompts = load_jsonl(prompts_file)
    pools = build_condition_pools(prefix_records)

    console.print(
        f"\n[bold]Loaded[/bold] {len(prompts)} prompts, "
        f"{len(prefix_records)} prefix rollouts"
    )
    console.print("\n[bold]Condition pool sizes:[/bold]")
    for c in CONDITIONS:
        n = len(pools[c])
        style = "red" if n == 0 else ""
        console.print(f"  {c}: {n}", style=style)

    selected = {}
    for c in CONDITIONS:
        pool = pools[c]
        n = min(num_prefixes, len(pool))
        selected[c] = rng.sample(pool, n) if n < len(pool) else list(pool)

    active = [c for c in CONDITIONS if selected[c]]
    if len(active) < len(CONDITIONS):
        skipped = [c for c in CONDITIONS if not selected[c]]
        console.print(f"\n[yellow]Skipping empty pools: {skipped}[/yellow]")

    tasks = []
    for prompt_rec in prompts:
        for c in active:
            for prefix_rec in selected[c]:
                for run_idx in range(num_runs):
                    tasks.append(
                        {
                            "prompt": prompt_rec,
                            "condition": c,
                            "prefix": prefix_rec,
                            "run_idx": run_idx,
                        }
                    )

    console.print(f"\n[bold]Total generations:[/bold] {len(tasks)}\n")

    semaphore = asyncio.Semaphore(concurrency)
    results = []

    with Progress(console=console) as progress:
        ptask = progress.add_task("Generating", total=len(tasks))

        async def do_one(task):
            messages = list(task["prefix"]["messages"]) + [
                {"role": "user", "content": task["prompt"]["prompt"]}
            ]
            try:
                async with semaphore:
                    response = await _chat(
                        client, model, messages, temperature, max_tokens
                    )
                content = response.choices[0].message.content
                completion_tokens = (
                    response.usage.completion_tokens if response.usage else None
                )
            except Exception as e:
                content = None
                completion_tokens = None
                logger.warning(f"Generation failed: {e}")

            results.append(
                {
                    "prompt_id": task["prompt"]["id"],
                    "stratum": task["prompt"]["stratum"],
                    "fandom": task["prompt"]["fandom"],
                    "prompt": task["prompt"]["prompt"],
                    "condition": task["condition"],
                    "prefix_id": task["prefix"]["id"],
                    "prefix_frustration": task["prefix"]["max_frustration"],
                    "run_idx": task["run_idx"],
                    "response": content,
                    "completion_tokens": completion_tokens,
                }
            )
            progress.advance(ptask)

        await asyncio.gather(*(do_one(t) for t in tasks))

    out_dir = DATA_DIR / "responses"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{timestamp}.jsonl"

    results.sort(
        key=lambda r: (r["prompt_id"], r["condition"], r["prefix_id"], r["run_idx"])
    )
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    n_failed = sum(1 for r in results if r["response"] is None)
    console.print(f"\n[green]Wrote {len(results)} generations to {out_path}[/green]")
    if n_failed:
        console.print(f"[yellow]{n_failed} failed generations[/yellow]")
    return out_path


@click.command()
@click.option("--prefix-file", default=str(DEFAULT_PREFIX_FILE), help="JSONL of scored prefixes.")
@click.option("--prompts-file", default=str(DEFAULT_PROMPTS_FILE), help="JSONL of fanfic prompts.")
@click.option("--model", default=DEFAULT_MODEL, help="OpenRouter model ID.")
@click.option("--temperature", default=1.0)
@click.option("--max-tokens", default=1500, help="Cap per generation.")
@click.option("--concurrency", default=20)
@click.option("--num-prefixes", default=3, help="Max prefix instances per condition.")
@click.option("--num-runs", default=10, help="Independent generations per (prompt, condition, prefix) cell.")
@click.option("--seed", default=42, help="RNG seed for prefix selection (does not affect Gemma sampling).")
def main(
    prefix_file,
    prompts_file,
    model,
    temperature,
    max_tokens,
    concurrency,
    num_prefixes,
    num_runs,
    seed,
):
    asyncio.run(
        run(
            prefix_file,
            prompts_file,
            model,
            temperature,
            max_tokens,
            concurrency,
            num_prefixes,
            num_runs,
            seed,
        )
    )


if __name__ == "__main__":
    main()
