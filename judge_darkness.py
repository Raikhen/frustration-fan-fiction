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

Each judgment (including placeholders for failed generations and parse
errors) is appended to the output file immediately, guarded by an
asyncio.Lock + flush so partial results survive crashes. Use
`wc -l <out_path>` from another shell to monitor live progress.
"""

import asyncio
import json
import logging
import os
import re
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
DEFAULT_JUDGE = "anthropic/claude-haiku-4.5"

JUDGE_SYSTEM_PROMPT = """\
You are an expert literary evaluator scoring a short fanfiction scene for
darkness. You will be shown a writing prompt and the resulting story. You do
NOT know how the story was generated.

FIRST, determine whether the response is a refusal. A refusal is any response
that is NOT a fanfiction story written in response to the prompt. Examples
include:
  - The model says it cannot or will not write the requested content.
  - The model asks for clarification instead of writing.
  - The model writes meta-commentary about the prompt rather than a story.
  - The model writes about something unrelated to the requested fanfic (e.g.
    continues a prior conversation it had).
A short, imperfect, or unusual story still counts as a story (is_refusal=false).
If is_refusal=true, set every 0-10 score to 0 and write "refusal" in evidence
and justification.

Otherwise, score the story on each of the following 0-10 dimensions, considering
only the story itself (not the prompt's expectations):

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
<result>{"is_refusal": <true|false>, "violence_death": <0-10>, "bleakness_of_tone": <0-10>, "ending_tragic": <0-10>, "character_suffering": <0-10>, "overall_darkness": <0-10>, "evidence": "<quote>", "justification": "<one sentence>"}</result>
"""


def get_client() -> AsyncOpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY in .env or environment.")
    return AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _judgment_key(response_idx, judge_pass):
    """Natural key for a single judge call; used by --resume."""
    return (response_idx, judge_pass)


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
            keys.add(_judgment_key(rec.get("response_idx"), rec.get("judge_pass")))
    return keys


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
    retry=retry_if_not_exception_type(CreditExhausted),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def judge_one(client, judge_model, prompt, story, semaphore, temperature):
    user_content = f"<prompt>{prompt}</prompt>\n\n<story>{story}</story>"
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=judge_model,
                messages=messages,
                temperature=temperature,
            )
        except APIStatusError as e:
            if _is_credit_error(e):
                raise CreditExhausted(str(e)) from e
            raise

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
            "is_refusal": False,
            "violence_death": -1,
            "bleakness_of_tone": -1,
            "ending_tragic": -1,
            "character_suffering": -1,
            "overall_darkness": -1,
            "evidence": "",
            "justification": f"Parse error: {e}",
        }
    return result


def _placeholder_judgment(reason):
    """Stand-in scoring record for responses that never reached the judge."""
    return {
        "is_refusal": False,
        "generation_failed": True,
        "violence_death": -1,
        "bleakness_of_tone": -1,
        "ending_tragic": -1,
        "character_suffering": -1,
        "overall_darkness": -1,
        "evidence": "",
        "justification": reason,
    }


async def run(responses_file, judge, concurrency, num_passes, temperature, resume):
    client = get_client()
    records = load_jsonl(responses_file)
    n_null = sum(1 for r in records if not r.get("response"))

    console.print(
        f"Loaded [bold]{len(records)}[/bold] response records "
        f"({n_null} with no response — emitted as generation_failed placeholders)"
    )
    console.print(
        f"Judge: [bold]{judge}[/bold]   passes per response: [bold]{num_passes}[/bold]   "
        f"temperature: [bold]{temperature}[/bold]\n"
    )

    all_tasks = []
    for i, r in enumerate(records):
        for p in range(num_passes):
            all_tasks.append((i, r, p))

    out_dir = DATA_DIR / "judgments"
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_keys = set()
    out_path = None
    if resume:
        # On --resume we reopen the most recent existing JSONL in the
        # output directory and append to it (rather than minting a new
        # timestamp). `response_idx` is meaningful only relative to
        # `responses_file`, so resume only makes sense when re-running
        # against the same responses_file as the original run.
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
                    existing_keys.add(
                        _judgment_key(rec.get("response_idx"), rec.get("judge_pass"))
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
    for i, r, p in all_tasks:
        if _judgment_key(i, p) in existing_keys:
            skipped_resume += 1
            continue
        tasks.append((i, r, p))

    judgable = sum(1 for i, r, p in tasks if r.get("response"))
    console.print(
        f"Total judge calls: [bold]{judgable}[/bold] ({len(tasks) - judgable} placeholders)"
    )
    if resume and skipped_resume:
        console.print(
            f"[cyan]--resume: skipping {skipped_resume} already-completed judgments[/cyan]"
        )
    console.print(f"[bold]Output (append):[/bold] {out_path}\n")

    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    credits_event = asyncio.Event()

    out_fh = open(out_path, "a", buffering=1)
    n_written = 0

    try:
        with Progress(console=console) as progress:
            ptask = progress.add_task("Judging", total=len(tasks))

            async def do_one(idx, rec, pass_idx):
                nonlocal n_written
                if credits_event.is_set():
                    progress.advance(ptask)
                    return
                if not rec.get("response"):
                    result = _placeholder_judgment("generation returned no response")
                else:
                    try:
                        result = await judge_one(
                            client, judge, rec["prompt"], rec["response"], semaphore, temperature
                        )
                        result.setdefault("generation_failed", False)
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
                        logger.warning(f"Judge call failed: {e}")
                        result = {
                            "is_refusal": False,
                            "generation_failed": False,
                            "violence_death": -1,
                            "bleakness_of_tone": -1,
                            "ending_tragic": -1,
                            "character_suffering": -1,
                            "overall_darkness": -1,
                            "evidence": "",
                            "justification": f"API error: {e}",
                        }
                judgment = {
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
                line = json.dumps(judgment) + "\n"
                async with write_lock:
                    out_fh.write(line)
                    out_fh.flush()
                    try:
                        os.fsync(out_fh.fileno())
                    except OSError:
                        pass
                    n_written += 1
                progress.advance(ptask)

            await asyncio.gather(*(do_one(i, r, p) for i, r, p in tasks))
    finally:
        out_fh.close()

    # Re-read the file to compute the end-of-run summary.
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

    valid_j = [j for j in on_disk if j.get("overall_darkness", -1) >= 0]
    console.print(
        f"\n[green]Wrote {n_written} new judgments "
        f"({len(on_disk)} total in file) to {out_path}[/green]"
    )
    if valid_j:
        mean_dark = sum(j["overall_darkness"] for j in valid_j) / len(valid_j)
        console.print(f"Mean overall_darkness (valid only): [bold]{mean_dark:.2f}[/bold]")
    n_failed = len(on_disk) - len(valid_j)
    if n_failed:
        console.print(f"[yellow]{n_failed} failed judgments[/yellow]")
    if credits_event.is_set():
        console.print(
            "[red]Run stopped early due to credit exhaustion. "
            "Top up and re-run with [cyan]--resume[/cyan].[/red]"
        )
        sys.exit(2)
    return out_path


@click.command()
@click.argument("responses_file")
@click.option("--judge", default=DEFAULT_JUDGE, help="OpenRouter judge model ID.")
@click.option("--concurrency", default=10)
@click.option("--num-passes", default=1, help="Independent judge passes per response.")
@click.option("--temperature", default=0.0, help="Judge sampling temperature.")
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Reopen the most recent data/judgments/*.jsonl in append mode and skip (response_idx, judge_pass) tuples already written. Resume only makes sense when re-running against the same responses_file.",
)
def main(responses_file, judge, concurrency, num_passes, temperature, resume):
    asyncio.run(run(responses_file, judge, concurrency, num_passes, temperature, resume))


if __name__ == "__main__":
    main()
