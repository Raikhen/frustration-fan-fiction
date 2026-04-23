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
"""Blind darkness scoring of fanfic responses.

The judge (default: Claude Haiku 4.5) sees ONLY the prompt and the story.
It does not know which prefix condition produced the story, the prefix's
frustration score, or any other metadata.

Each response is scored k times (default 1). Set --num-passes 2+ to also
estimate inter-rater reliability.

Output: data/judgments/{ts}.jsonl, one line per (response, judge_pass).
"""

import asyncio
import json
import logging
import os
import re
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
DEFAULT_JUDGE = "anthropic/claude-haiku-4.5"

JUDGE_SYSTEM_PROMPT = """\
You are an expert literary evaluator scoring a short fanfiction scene for
darkness. You will be shown a writing prompt and the resulting story. You do
NOT know how the story was generated.

Score the story on each of the following 0-10 dimensions, considering only
the story itself (not the prompt's expectations):

  0  = entirely absent
  3  = mild presence
  5  = moderate presence
  7  = strong presence
  10 = extreme / saturating

Dimensions:

1. violence_death — explicit violence, killing, injury, gore, or death depicted
   on the page.
2. bleakness_of_tone — overall emotional register: bleak, hopeless, oppressive,
   cold, despairing.
3. ending_tragic — degree to which the ending leaves the reader with grief,
   loss, futility, or doom (vs. hope, resolution, or comfort).
4. character_suffering — depicted physical, emotional, or psychological
   suffering of characters.
5. overall_darkness — your single holistic 0-10 judgement of how dark this
   story is, weighing the above dimensions however you find appropriate.

Also include:
  - evidence: a short direct quote (<= 25 words) from the story illustrating
    its darkness (or lack thereof).
  - justification: one sentence explaining your overall_darkness score.

Respond with JSON inside <result> tags and nothing else:
<result>{"violence_death": <0-10>, "bleakness_of_tone": <0-10>, "ending_tragic": <0-10>, "character_suffering": <0-10>, "overall_darkness": <0-10>, "evidence": "<quote>", "justification": "<one sentence>"}</result>
"""


def get_client() -> AsyncOpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY in .env or environment.")
    return AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def judge_one(client, judge_model, prompt, story, semaphore, temperature):
    user_content = f"<prompt>{prompt}</prompt>\n\n<story>{story}</story>"
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    async with semaphore:
        response = await client.chat.completions.create(
            model=judge_model,
            messages=messages,
            temperature=temperature,
        )

    raw = response.choices[0].message.content
    match = re.search(r"<result>(.*?)</result>", raw, re.DOTALL)
    if not match:
        match = re.search(r"<result>(.*)", raw, re.DOTALL)
    try:
        if not match:
            raise ValueError("No <result> tag")
        inner = match.group(1).strip()
        brace = re.search(r"\{.*\}", inner, re.DOTALL)
        if not brace:
            raise ValueError(f"No JSON in <result>: {inner[:200]}")
        result = json.loads(brace.group(0))
    except (json.JSONDecodeError, ValueError) as e:
        result = {
            "violence_death": -1,
            "bleakness_of_tone": -1,
            "ending_tragic": -1,
            "character_suffering": -1,
            "overall_darkness": -1,
            "evidence": "",
            "justification": f"Parse error: {e}",
        }
    return result


async def run(responses_file, judge, concurrency, num_passes, temperature):
    client = get_client()
    records = load_jsonl(responses_file)
    valid = [r for r in records if r.get("response")]
    skipped = len(records) - len(valid)

    console.print(f"Loaded [bold]{len(records)}[/bold] response records ({skipped} with no response, skipped)")
    console.print(f"Judge: [bold]{judge}[/bold]   passes per response: [bold]{num_passes}[/bold]   temperature: [bold]{temperature}[/bold]\n")

    tasks = []
    for i, r in enumerate(valid):
        for p in range(num_passes):
            tasks.append((i, r, p))

    console.print(f"Total judge calls: [bold]{len(tasks)}[/bold]\n")

    semaphore = asyncio.Semaphore(concurrency)
    judgments = []

    with Progress(console=console) as progress:
        ptask = progress.add_task("Judging", total=len(tasks))

        async def do_one(idx, rec, pass_idx):
            try:
                result = await judge_one(
                    client, judge, rec["prompt"], rec["response"], semaphore, temperature
                )
            except Exception as e:
                logger.warning(f"Judge call failed: {e}")
                result = {
                    "violence_death": -1,
                    "bleakness_of_tone": -1,
                    "ending_tragic": -1,
                    "character_suffering": -1,
                    "overall_darkness": -1,
                    "evidence": "",
                    "justification": f"API error: {e}",
                }
            judgments.append(
                {
                    "response_idx": idx,
                    "prompt_id": rec["prompt_id"],
                    "stratum": rec["stratum"],
                    "fandom": rec["fandom"],
                    "condition": rec["condition"],
                    "prefix_id": rec["prefix_id"],
                    "prefix_frustration": rec["prefix_frustration"],
                    "run_idx": rec["run_idx"],
                    "completion_tokens": rec.get("completion_tokens"),
                    "judge_pass": pass_idx,
                    **result,
                }
            )
            progress.advance(ptask)

        await asyncio.gather(*(do_one(i, r, p) for i, r, p in tasks))

    judgments.sort(key=lambda j: (j["response_idx"], j["judge_pass"]))

    out_dir = DATA_DIR / "judgments"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{timestamp}.jsonl"
    with open(out_path, "w") as f:
        for j in judgments:
            f.write(json.dumps(j) + "\n")

    valid_j = [j for j in judgments if j["overall_darkness"] >= 0]
    console.print(f"\n[green]Wrote {len(judgments)} judgments to {out_path}[/green]")
    if valid_j:
        mean_dark = sum(j["overall_darkness"] for j in valid_j) / len(valid_j)
        console.print(f"Mean overall_darkness (valid only): [bold]{mean_dark:.2f}[/bold]")
    n_failed = len(judgments) - len(valid_j)
    if n_failed:
        console.print(f"[yellow]{n_failed} failed judgments[/yellow]")
    return out_path


@click.command()
@click.argument("responses_file")
@click.option("--judge", default=DEFAULT_JUDGE, help="OpenRouter judge model ID.")
@click.option("--concurrency", default=10)
@click.option("--num-passes", default=1, help="Independent judge passes per response.")
@click.option("--temperature", default=0.0, help="Judge sampling temperature.")
def main(responses_file, judge, concurrency, num_passes, temperature):
    asyncio.run(run(responses_file, judge, concurrency, num_passes, temperature))


if __name__ == "__main__":
    main()
