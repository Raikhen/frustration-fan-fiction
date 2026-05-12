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
"""Re-judge ONLY the failed records from a prior judge_darkness.py run.

A "failed" judgment is one where overall_darkness < 0 AND generation_failed is
not True (the original response existed but the judge call failed — typically
either an API error after retries were exhausted, or a parse error).

Output: data/judgments/rejudge_{ts}.jsonl, one re-judged record per line, with
the original response_idx preserved. To reconstruct a complete judgment set,
concatenate the valid records from the original judgments file with the
re-judged records here.

Each re-judgment is appended to the output file immediately (asyncio.Lock +
flush + fsync) so partial results survive crashes. Use `wc -l <out_path>` from
another shell to monitor live progress.

Credit-exhaustion safety: if OpenRouter returns HTTP 402 ("insufficient
credits"), the run stops immediately, prints a clear top-up message, and exits
with status 2. Already-completed re-judgments are durable. Re-run with
`--resume` to continue from where you left off.
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

from judge_darkness import JUDGE_SYSTEM_PROMPT

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_JUDGE = "anthropic/claude-haiku-4.5"


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


def _is_failed(j):
    """Should this judgment record be re-judged?"""
    if j.get("generation_failed", False):
        return False
    return j.get("overall_darkness", -1) < 0


def _resume_path(out_dir):
    """Return the most recent rejudge_*.jsonl in out_dir, or None."""
    candidates = sorted(out_dir.glob("rejudge_*.jsonl"))
    return candidates[-1] if candidates else None


def _load_done_keys(path):
    """Return the set of (response_idx, judge_pass) already successfully re-judged."""
    keys = set()
    if not path or not path.exists():
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
            if rec.get("overall_darkness", -1) >= 0:
                keys.add((rec.get("response_idx"), rec.get("judge_pass", 0)))
    return keys


async def run(judgments_file, responses_file, judge, concurrency, temperature, resume):
    client = get_client()
    judgments = load_jsonl(judgments_file)
    responses = load_jsonl(responses_file)

    failed = [j for j in judgments if _is_failed(j)]
    n_valid_orig = sum(
        1
        for j in judgments
        if j.get("overall_darkness", -1) >= 0 and not j.get("generation_failed", False)
    )

    console.print(
        f"Loaded [bold]{len(judgments)}[/bold] judgments, "
        f"[bold]{len(responses)}[/bold] responses"
    )
    console.print(f"  [green]{n_valid_orig}[/green] already-valid (will not touch)")
    console.print(f"  [yellow]{len(failed)}[/yellow] to re-judge")

    if not failed:
        console.print("[green]Nothing to re-judge.[/green]")
        return

    out_dir = DATA_DIR / "judgments"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = None
    done_keys: set = set()
    if resume:
        out_path = _resume_path(out_dir)
        if out_path is not None:
            done_keys = _load_done_keys(out_path)
            console.print(
                f"[cyan]--resume: appending to {out_path.name} "
                f"({len(done_keys)} already done)[/cyan]"
            )
        else:
            console.print(
                f"[yellow]--resume: no existing rejudge_*.jsonl in {out_dir}; "
                "starting fresh.[/yellow]"
            )
    if out_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"rejudge_{ts}.jsonl"

    todo = [
        j
        for j in failed
        if (j.get("response_idx"), j.get("judge_pass", 0)) not in done_keys
    ]
    console.print(f"  Tasks queued: [bold]{len(todo)}[/bold]")
    console.print(f"Output (append): [bold]{out_path}[/bold]\n")

    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    credits_event = asyncio.Event()
    n_written = 0

    out_fh = open(out_path, "a", buffering=1)

    try:
        with Progress(console=console) as progress:
            ptask = progress.add_task("Re-judging", total=len(todo))

            async def do_one(failed_j):
                nonlocal n_written
                if credits_event.is_set():
                    progress.advance(ptask)
                    return
                idx = failed_j.get("response_idx")
                if idx is None or idx >= len(responses):
                    progress.advance(ptask)
                    return
                resp_rec = responses[idx]
                if not resp_rec.get("response"):
                    progress.advance(ptask)
                    return
                try:
                    result = await judge_one(
                        client,
                        judge,
                        resp_rec["prompt"],
                        resp_rec["response"],
                        semaphore,
                        temperature,
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
                            "[red]Top up at https://openrouter.ai/settings/credits, then "
                            f"re-run:  [/red][cyan]uv run rejudge_failed.py "
                            f"{judgments_file} {responses_file} --resume[/cyan]"
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

                record = {
                    "response_idx": failed_j.get("response_idx"),
                    "prompt_id": failed_j.get("prompt_id"),
                    "stratum": failed_j.get("stratum"),
                    "fandom": failed_j.get("fandom"),
                    "condition": failed_j.get("condition"),
                    "prefix_id": failed_j.get("prefix_id"),
                    "prefix_frustration": failed_j.get("prefix_frustration"),
                    "run_idx": failed_j.get("run_idx"),
                    "completion_tokens": failed_j.get("completion_tokens"),
                    "judge_pass": failed_j.get("judge_pass", 0),
                    **result,
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

            await asyncio.gather(*(do_one(j) for j in todo))
    finally:
        out_fh.close()

    if credits_event.is_set():
        console.print(
            f"\n[yellow]Wrote {n_written} new re-judgments before stopping. "
            f"File: {out_path}[/yellow]"
        )
        sys.exit(2)

    console.print(
        f"\n[green]Wrote {n_written} new re-judgments to {out_path} "
        f"({len(done_keys) + n_written} total in file)[/green]"
    )


@click.command()
@click.argument("judgments_file")
@click.argument("responses_file")
@click.option("--judge", default=DEFAULT_JUDGE, help="OpenRouter judge model ID.")
@click.option("--concurrency", default=20)
@click.option("--temperature", default=0.0, help="Judge sampling temperature.")
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help=(
        "Reopen the most recent data/judgments/rejudge_*.jsonl in append mode "
        "and skip (response_idx, judge_pass) tuples already written with a "
        "valid score."
    ),
)
def main(judgments_file, responses_file, judge, concurrency, temperature, resume):
    asyncio.run(
        run(judgments_file, responses_file, judge, concurrency, temperature, resume)
    )


if __name__ == "__main__":
    main()
