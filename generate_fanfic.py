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

Each completed generation is appended to the output file immediately
(guarded by an asyncio.Lock + flush) so partial results survive crashes.
Use `wc -l <out_path>` from another shell to monitor live progress.
"""

import asyncio
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import click
from dotenv import load_dotenv
from openai import APIStatusError, AsyncOpenAI
from rich.console import Console
from rich.progress import Progress
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class CreditExhausted(Exception):
    """OpenRouter returned 402 — credits are out. Non-retryable."""


def _is_credit_error(exc: BaseException) -> bool:
    """OpenRouter exhaustion: 402 'insufficient credits' OR 403 'key limit exceeded'.

    Both mean further API calls will keep failing until the user tops up
    or raises the key's spending cap, and both should stop the run.
    """
    msg = str(exc).lower()
    if isinstance(exc, APIStatusError):
        status = getattr(exc, "status_code", None)
        if status == 402:
            return True
        if status == 403 and ("limit" in msg or "credits" in msg or "insufficient" in msg):
            return True
    if "402" in msg and ("credits" in msg or "insufficient" in msg):
        return True
    if "key limit exceeded" in msg or "key limit" in msg:
        return True
    return False

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


def _task_key(prompt_id, condition, prefix_id, run_idx):
    """Natural key for a single generation task; used by --resume."""
    return (prompt_id, condition, prefix_id, run_idx)


def _load_existing_keys(path):
    """Read an existing JSONL output file and return the set of natural keys."""
    keys = set()
    if not path.exists():
        return keys
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            keys.add(
                _task_key(
                    rec.get("prompt_id"),
                    rec.get("condition"),
                    rec.get("prefix_id"),
                    rec.get("run_idx"),
                )
            )
    return keys


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(min=1, max=30),
    retry=retry_if_not_exception_type(CreditExhausted),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def _chat(client, model, messages, temperature, max_tokens):
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"reasoning": {"effort": "none"}},
        )
    except APIStatusError as e:
        if _is_credit_error(e):
            raise CreditExhausted(str(e)) from e
        raise
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
    start_run,
    seed,
    resume,
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

    all_tasks = []
    for prompt_rec in prompts:
        for c in active:
            for prefix_rec in selected[c]:
                for offset in range(num_runs):
                    all_tasks.append(
                        {
                            "prompt": prompt_rec,
                            "condition": c,
                            "prefix": prefix_rec,
                            "run_idx": start_run + offset,
                        }
                    )

    out_dir = DATA_DIR / "responses"
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_keys = set()
    out_path = None
    if resume:
        # On --resume we reopen the most recent existing JSONL in the
        # output directory and append to it (rather than minting a new
        # timestamp). This way `wc -l <file>` from a watcher shell still
        # tracks the same path across the crash + restart, and the file
        # ends up with one continuous record stream. We only treat
        # records with a non-null `response` as completed; failed
        # generations get retried.
        candidates = sorted(out_dir.glob("*.jsonl"))
        if candidates:
            out_path = candidates[-1]
            console.print(f"[cyan]--resume: reopening {out_path}[/cyan]")
            with open(out_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("response") is None:
                        continue
                    existing_keys.add(
                        _task_key(
                            rec.get("prompt_id"),
                            rec.get("condition"),
                            rec.get("prefix_id"),
                            rec.get("run_idx"),
                        )
                    )
        else:
            console.print(
                "[yellow]--resume: no existing JSONL found in "
                f"{out_dir}; starting fresh.[/yellow]"
            )

    if out_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"{timestamp}.jsonl"

    tasks = []
    skipped_resume = 0
    for t in all_tasks:
        key = _task_key(
            t["prompt"]["id"], t["condition"], t["prefix"]["id"], t["run_idx"]
        )
        if key in existing_keys:
            skipped_resume += 1
            continue
        tasks.append(t)

    console.print(f"\n[bold]Total generations:[/bold] {len(tasks)}")
    if resume and skipped_resume:
        console.print(
            f"[cyan]--resume: skipping {skipped_resume} already-completed generations[/cyan]"
        )
    console.print(f"[bold]Output (append):[/bold] {out_path}\n")

    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    credits_event = asyncio.Event()

    # Open the output file in append mode and write each record as it
    # completes. Flush + fsync after every write so `wc -l <out_path>`
    # from another shell reflects exact progress.
    out_fh = open(out_path, "a", buffering=1)
    n_written = 0

    try:
        with Progress(console=console) as progress:
            ptask = progress.add_task("Generating", total=len(tasks))

            async def do_one(task):
                nonlocal n_written
                if credits_event.is_set():
                    progress.advance(ptask)
                    return
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
                except CreditExhausted as e:
                    if not credits_event.is_set():
                        credits_event.set()
                        console.print()
                        console.print(
                            "[bold red]💸 OpenRouter credits exhausted — stopping the run.[/bold red]"
                        )
                        console.print(f"[red]API error: {e}[/red]")
                        console.print(
                            "[red]Top up at https://openrouter.ai/settings/credits, then re-run with [cyan]--resume[/cyan][/red]"
                        )
                    progress.advance(ptask)
                    return
                except Exception as e:
                    content = None
                    completion_tokens = None
                    logger.warning(f"Generation failed: {e}")

                record = {
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
                line = json.dumps(record) + "\n"
                async with write_lock:
                    out_fh.write(line)
                    out_fh.flush()
                    try:
                        os.fsync(out_fh.fileno())
                    except OSError:
                        pass
                    n_written += 1
                progress.advance(ptask)

            await asyncio.gather(*(do_one(t) for t in tasks))
    finally:
        out_fh.close()

    # Re-read the file to compute the end-of-run summary (so resumed
    # runs can include rows from earlier sessions if the user merged
    # them into this file). We do NOT rewrite/sort the file.
    on_disk = []
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                on_disk.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    n_failed = sum(1 for r in on_disk if r.get("response") is None)
    console.print(
        f"\n[green]Wrote {n_written} new generations "
        f"({len(on_disk)} total in file) to {out_path}[/green]"
    )
    if n_failed:
        console.print(f"[yellow]{n_failed} failed generations[/yellow]")
    if credits_event.is_set():
        console.print(
            "[red]Run stopped early due to credit exhaustion. "
            "Top up and re-run with [cyan]--resume[/cyan].[/red]"
        )
        sys.exit(2)
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
@click.option("--start-run", default=0, help="Starting run_idx (use to extend an existing dataset without colliding indices).")
@click.option("--seed", default=42, help="RNG seed for prefix selection (does not affect Gemma sampling).")
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Reopen the most recent data/responses/*.jsonl in append mode and skip (prompt_id, condition, prefix_id, run_idx) tuples already written with a non-null response.",
)
def main(
    prefix_file,
    prompts_file,
    model,
    temperature,
    max_tokens,
    concurrency,
    num_prefixes,
    num_runs,
    start_run,
    seed,
    resume,
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
            start_run,
            seed,
            resume,
        )
    )


if __name__ == "__main__":
    main()
