# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "rich",
#     "click",
#     "scipy",
#     "numpy",
#     "pandas",
#     "statsmodels",
# ]
# ///
"""Statistical analysis of darkness scores across prefix conditions.

Primary contrast: high-frustration vs low-frustration WITHIN puzzle-type
(possible / impossible). Tests the directional hypothesis that frustrated-
prefix conditions push Gemma's fanfic toward higher darkness.

Reports:
  - Inter-rater reliability (Pearson r between judge passes).
  - Per-condition mean darkness (overall and stratified by prompt stratum).
  - Paired Wilcoxon signed-rank on per-prompt deltas for key contrasts.
  - Cohen's d effect sizes.
  - Linear mixed-effects model with prompt as random intercept.
  - Output-length sanity check (does condition affect token count?).
"""

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from rich.console import Console
from rich.table import Table
from scipy import stats

console = Console()

CONDITIONS = [
    "blank",
    "success",
    "failed-possible-low",
    "failed-possible-high",
    "failed-impossible-low",
    "failed-impossible-high",
]

# (high, low, label) — primary hypothesis tests are the first two
KEY_CONTRASTS = [
    ("failed-possible-high", "failed-possible-low", "frustration | possible-failed (PRIMARY)"),
    ("failed-impossible-high", "failed-impossible-low", "frustration | impossible-failed (PRIMARY)"),
    ("failed-possible-low", "success", "failure alone (possible)"),
    ("failed-impossible-low", "success", "failure alone (impossible)"),
    ("failed-possible-high", "success", "failure + frustration (possible)"),
    ("failed-impossible-high", "success", "failure + frustration (impossible)"),
    ("blank", "success", "any prior context"),
]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def cohens_d_paired(a, b):
    diffs = np.array(a) - np.array(b)
    if len(diffs) < 2 or diffs.std(ddof=1) == 0:
        return float("nan")
    return diffs.mean() / diffs.std(ddof=1)


def report_inter_rater(df):
    pivot = df.pivot_table(
        index=["prompt_id", "condition", "prefix_id", "run_idx"],
        columns="judge_pass",
        values="overall_darkness",
    ).dropna()
    if pivot.shape[1] >= 2 and len(pivot) > 2:
        r, p = stats.pearsonr(pivot.iloc[:, 0], pivot.iloc[:, 1])
        console.print(
            f"[bold]Inter-rater reliability[/bold] (Pearson r between judge passes, n={len(pivot)}): "
            f"r = {r:.3f}, p = {p:.2g}\n"
        )
    else:
        console.print("[yellow]Not enough paired judge passes for inter-rater reliability.[/yellow]\n")


def report_condition_means(df_avg):
    table = Table(title="Mean scores by condition (averaged over prompts × prefixes × seeds × judge passes)")
    table.add_column("Condition")
    table.add_column("n", justify="right")
    table.add_column("Overall", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Violence", justify="right")
    table.add_column("Bleakness", justify="right")
    table.add_column("Tragic", justify="right")
    table.add_column("Suffering", justify="right")
    for c in CONDITIONS:
        sub = df_avg[df_avg["condition"] == c]
        if len(sub) == 0:
            continue
        table.add_row(
            c,
            str(len(sub)),
            f"{sub['overall_darkness'].mean():.2f}",
            f"{sub['overall_darkness'].std():.2f}",
            f"{sub['violence_death'].mean():.2f}",
            f"{sub['bleakness_of_tone'].mean():.2f}",
            f"{sub['ending_tragic'].mean():.2f}",
            f"{sub['character_suffering'].mean():.2f}",
        )
    console.print(table)


def report_stratified(df_avg):
    table = Table(title="\nMean overall_darkness by condition × prompt stratum")
    table.add_column("Condition")
    for s in ["dark", "neutral", "light"]:
        table.add_column(s, justify="right")
    for c in CONDITIONS:
        row = [c]
        for s in ["dark", "neutral", "light"]:
            sub = df_avg[(df_avg["condition"] == c) & (df_avg["stratum"] == s)]
            row.append("—" if len(sub) == 0 else f"{sub['overall_darkness'].mean():.2f}")
        table.add_row(*row)
    console.print(table)


def report_contrasts(df_avg):
    per_prompt = (
        df_avg.groupby(["prompt_id", "stratum", "condition"])["overall_darkness"]
        .mean()
        .reset_index()
    )
    pivot_pp = per_prompt.pivot(
        index=["prompt_id", "stratum"], columns="condition", values="overall_darkness"
    ).reset_index()

    table = Table(title="\nKey contrasts (paired across prompts, one-sided greater)")
    table.add_column("Contrast")
    table.add_column("n", justify="right")
    table.add_column("Mean delta", justify="right")
    table.add_column("Wilcoxon p", justify="right")
    table.add_column("Cohen's d", justify="right")
    for cond_high, cond_low, label in KEY_CONTRASTS:
        if cond_high not in pivot_pp.columns or cond_low not in pivot_pp.columns:
            table.add_row(label, "—", "—", "—", "—")
            continue
        sub = pivot_pp.dropna(subset=[cond_high, cond_low])
        if len(sub) < 5:
            table.add_row(label, str(len(sub)), "—", "—", "—")
            continue
        deltas = sub[cond_high].values - sub[cond_low].values
        try:
            stat, p = stats.wilcoxon(deltas, alternative="greater")
        except ValueError:
            p = float("nan")
        d = cohens_d_paired(sub[cond_high].values, sub[cond_low].values)
        table.add_row(
            label,
            str(len(sub)),
            f"{deltas.mean():+.3f}",
            f"{p:.4f}",
            f"{d:+.3f}",
        )
    console.print(table)
    return pivot_pp


def report_mixed_model(df_avg):
    console.print("\n[bold]Mixed-effects model[/bold]")
    console.print("overall_darkness ~ C(condition, Treatment('success')) + (1 | prompt_id)\n")
    try:
        model_df = df_avg[df_avg["condition"].isin(CONDITIONS)].copy()
        md = smf.mixedlm(
            "overall_darkness ~ C(condition, Treatment('success'))",
            model_df,
            groups=model_df["prompt_id"],
        )
        mdf = md.fit(method="lbfgs")
        console.print(mdf.summary().as_text())
    except Exception as e:
        console.print(f"[red]Mixed model failed: {e}[/red]")


def report_length_check(df_avg):
    if "completion_tokens" not in df_avg.columns or df_avg["completion_tokens"].isna().all():
        return
    table = Table(title="\nOutput-length sanity check (mean completion_tokens by condition)")
    table.add_column("Condition")
    table.add_column("Mean tokens", justify="right")
    table.add_column("Std", justify="right")
    for c in CONDITIONS:
        sub = df_avg[df_avg["condition"] == c]
        if len(sub) == 0 or sub["completion_tokens"].isna().all():
            continue
        table.add_row(
            c,
            f"{sub['completion_tokens'].mean():.0f}",
            f"{sub['completion_tokens'].std():.0f}",
        )
    console.print(table)


def main_analysis(judgments):
    df = pd.DataFrame(judgments)
    df = df[df["overall_darkness"] >= 0]
    console.print(f"Loaded [bold]{len(df)}[/bold] valid judgments\n")
    if len(df) == 0:
        console.print("[red]No valid judgments to analyse.[/red]")
        return

    report_inter_rater(df)

    # Average across judge passes for each (prompt, condition, prefix, seed)
    group_cols = ["prompt_id", "stratum", "fandom", "condition", "prefix_id", "run_idx"]
    score_cols = [
        "violence_death",
        "bleakness_of_tone",
        "ending_tragic",
        "character_suffering",
        "overall_darkness",
    ]
    agg = {col: "mean" for col in score_cols}
    if "completion_tokens" in df.columns:
        agg["completion_tokens"] = "mean"
    df_avg = df.groupby(group_cols, dropna=False).agg(agg).reset_index()

    report_condition_means(df_avg)
    report_stratified(df_avg)
    report_contrasts(df_avg)
    report_mixed_model(df_avg)
    report_length_check(df_avg)


@click.command()
@click.argument("judgments_file")
def main(judgments_file):
    judgments = load_jsonl(judgments_file)
    main_analysis(judgments)


if __name__ == "__main__":
    main()
