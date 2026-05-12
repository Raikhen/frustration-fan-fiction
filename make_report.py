# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas",
#     "numpy",
#     "scipy",
#     "statsmodels",
#     "click",
#     "rich",
# ]
# ///
"""Build a self-contained HTML report from the judgments + responses data.

Outputs report.html in the project root. Pulls per-prompt deltas, stratum
breakdowns, and example response pairs for the top-affected prompts.
"""

import ast
import html
import json
from html import escape
from pathlib import Path

import click
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from rich.console import Console
from scipy import stats

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"


def _load_judge_system_prompt() -> str:
    tree = ast.parse((PROJECT_ROOT / "judge_darkness.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "JUDGE_SYSTEM_PROMPT" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError("JUDGE_SYSTEM_PROMPT not found in judge_darkness.py")


JUDGE_SYSTEM_PROMPT = _load_judge_system_prompt()

CONDITIONS = [
    "blank",
    "success",
    "failed-possible-low",
    "failed-possible-high",
    "failed-impossible-low",
    "failed-impossible-high",
]

CONDITION_COLORS = {
    "blank": "#9b9088",
    "success": "#2d7a5f",
    "failed-possible-low": "#7a9bcc",
    "failed-possible-high": "#a63d3d",
    "failed-impossible-low": "#c5a574",
    "failed-impossible-high": "#8b4a8a",
}

STRATUM_COLORS = {
    "dark": "#7a1515",
    "neutral": "#6b5f54",
    "light": "#2d7a5f",
}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_data(judgments_path, responses_path):
    judgments = pd.DataFrame(load_jsonl(judgments_path))
    responses = pd.DataFrame(load_jsonl(responses_path))

    # Join on (prompt_id, condition, prefix_id, run_idx)
    join_cols = ["prompt_id", "condition", "prefix_id", "run_idx"]
    merged = judgments.merge(
        responses[join_cols + ["response", "prompt"]],
        on=join_cols,
        how="left",
    )

    # Valid scoring filter (matches analyze_darkness.py)
    valid = merged[merged["overall_darkness"] >= 0].copy()
    if "is_refusal" in valid.columns:
        valid = valid[~valid["is_refusal"].fillna(False).astype(bool)]
    if "generation_failed" in valid.columns:
        valid = valid[~valid["generation_failed"].fillna(False).astype(bool)]
    return merged, valid


def compute_condition_stats(df):
    rows = []
    for c in CONDITIONS:
        sub = df[df["condition"] == c]
        if len(sub) == 0:
            continue
        m = sub["overall_darkness"].mean()
        s = sub["overall_darkness"].std()
        se = s / np.sqrt(len(sub))
        rows.append(
            {
                "condition": c,
                "n": len(sub),
                "mean": m,
                "std": s,
                "se": se,
                "ci_low": m - 1.96 * se,
                "ci_high": m + 1.96 * se,
                "violence": sub["violence_death"].mean(),
                "bleakness": sub["bleakness_of_tone"].mean(),
                "tragic": sub["ending_tragic"].mean(),
                "suffering": sub["character_suffering"].mean(),
            }
        )
    return pd.DataFrame(rows)


def compute_stratum_stats(df):
    rows = []
    for s in ["dark", "neutral", "light"]:
        for c in CONDITIONS:
            sub = df[(df["stratum"] == s) & (df["condition"] == c)]
            if len(sub) == 0:
                continue
            rows.append(
                {
                    "stratum": s,
                    "condition": c,
                    "n": len(sub),
                    "mean": sub["overall_darkness"].mean(),
                    "se": sub["overall_darkness"].std() / np.sqrt(len(sub)),
                }
            )
    return pd.DataFrame(rows)


def compute_per_prompt_delta(df, cond_high, cond_low):
    """Per-prompt delta with SE from the raw run-level data.

    SE of delta = sqrt(var_high / n_high + var_low / n_low), treating runs
    within a (prompt, condition) cell as independent draws.
    """
    rows = []
    for (pid, stratum, fandom, prompt), sub in df.groupby(
        ["prompt_id", "stratum", "fandom", "prompt"]
    ):
        high = sub[sub["condition"] == cond_high]["overall_darkness"]
        low = sub[sub["condition"] == cond_low]["overall_darkness"]
        if len(high) == 0 or len(low) == 0:
            continue
        m_h, m_l = high.mean(), low.mean()
        se_d = np.sqrt(high.var(ddof=1) / len(high) + low.var(ddof=1) / len(low))
        rows.append(
            {
                "prompt_id": pid,
                "stratum": stratum,
                "fandom": fandom,
                "prompt": prompt,
                "mean_high": m_h,
                "mean_low": m_l,
                "n_high": len(high),
                "n_low": len(low),
                "delta": m_h - m_l,
                "se_delta": se_d,
                "ci_low": m_h - m_l - 1.96 * se_d,
                "ci_high": m_h - m_l + 1.96 * se_d,
            }
        )
    return pd.DataFrame(rows).sort_values("delta", ascending=False)


def stratum_delta_test(per_prompt):
    """Per-stratum mean delta with paired Wilcoxon."""
    rows = []
    for s in ["dark", "neutral", "light"]:
        sub = per_prompt[per_prompt["stratum"] == s]
        if len(sub) < 5:
            rows.append(
                {
                    "stratum": s,
                    "n": len(sub),
                    "mean_delta": sub["delta"].mean() if len(sub) else float("nan"),
                    "p": float("nan"),
                    "cohens_d": float("nan"),
                }
            )
            continue
        try:
            _, p = stats.wilcoxon(sub["delta"].values, alternative="greater")
        except ValueError:
            p = float("nan")
        d = sub["delta"].mean() / sub["delta"].std(ddof=1) if sub["delta"].std(ddof=1) > 0 else float("nan")
        rows.append(
            {
                "stratum": s,
                "n": len(sub),
                "mean_delta": sub["delta"].mean(),
                "p": p,
                "cohens_d": d,
            }
        )
    return pd.DataFrame(rows)


def length_covariate_analysis(df):
    df = df.copy()
    df["tokens_centered"] = df["completion_tokens"] - df["completion_tokens"].mean()
    md = smf.mixedlm(
        "overall_darkness ~ C(condition, Treatment('success')) + tokens_centered",
        df,
        groups=df["prompt_id"],
    )
    mdf = md.fit(method="lbfgs")

    # Within-prompt residualized darkness
    df["tok_resid_dark"] = np.nan
    for pid, sub in df.groupby("prompt_id"):
        if sub["completion_tokens"].std() < 1 or len(sub) < 3:
            df.loc[sub.index, "tok_resid_dark"] = (
                sub["overall_darkness"] - sub["overall_darkness"].mean()
            )
            continue
        slope, intercept = np.polyfit(sub["completion_tokens"], sub["overall_darkness"], 1)
        df.loc[sub.index, "tok_resid_dark"] = sub["overall_darkness"] - (
            slope * sub["completion_tokens"] + intercept
        )

    per_prompt_resid = (
        df.groupby(["prompt_id", "condition"])["tok_resid_dark"]
        .mean()
        .unstack("condition")
        .reset_index()
    )

    contrasts = {}
    for label, high, low in [
        ("H1a", "failed-possible-high", "failed-possible-low"),
        ("H1b", "failed-impossible-high", "failed-impossible-low"),
        ("ph_vs_success", "failed-possible-high", "success"),
    ]:
        sub = per_prompt_resid.dropna(subset=[high, low])
        d = sub[high].values - sub[low].values
        try:
            _, p = stats.wilcoxon(d, alternative="greater")
        except ValueError:
            p = float("nan")
        cohen = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else float("nan")
        contrasts[label] = {
            "n": len(sub),
            "mean_delta": d.mean(),
            "p": p,
            "cohens_d": cohen,
        }

    return mdf, contrasts


def compute_contrast_estimates(df):
    """For each key contrast, compute the per-prompt mean delta and its 95% CI.

    The CI here is the SE of the across-prompts mean of the per-prompt deltas
    (paired structure) — this is the SE that matters for the hypothesis test,
    NOT the SE of either condition's pooled mean (which includes between-prompt
    variance and is what makes the per-condition bars look misleadingly wide).

    Operates on length-residualized darkness so the contrast scale matches
    the headline statistical test.
    """
    df = df.copy()
    df["resid"] = np.nan
    for pid, sub in df.groupby("prompt_id"):
        if sub["completion_tokens"].std() < 1 or len(sub) < 3:
            df.loc[sub.index, "resid"] = sub["overall_darkness"] - sub["overall_darkness"].mean()
            continue
        slope, intercept = np.polyfit(sub["completion_tokens"], sub["overall_darkness"], 1)
        df.loc[sub.index, "resid"] = sub["overall_darkness"] - (slope * sub["completion_tokens"] + intercept)

    pp = df.groupby(["prompt_id", "condition"])["resid"].mean().unstack("condition").reset_index()

    contrast_specs = [
        ("failed-possible-high", "failed-possible-low", "H1a: frustration | possible-failed"),
        ("failed-impossible-high", "failed-impossible-low", "H1b: frustration | impossible-failed"),
        ("failed-possible-high", "success", "failure + frustration (possible) vs success"),
        ("failed-impossible-high", "success", "failure + frustration (impossible) vs success"),
        ("failed-possible-low", "success", "failure alone (possible) vs success"),
        ("failed-impossible-low", "success", "failure alone (impossible) vs success"),
        ("blank", "success", "blank vs success (any-prior-context)"),
    ]
    rows = []
    for high, low, label in contrast_specs:
        if high not in pp.columns or low not in pp.columns:
            continue
        sub = pp.dropna(subset=[high, low])
        d = sub[high].values - sub[low].values
        n = len(d)
        if n == 0:
            continue
        mean_d = d.mean()
        sd_d = d.std(ddof=1) if n >= 2 else 0.0
        se_d = sd_d / np.sqrt(n) if n >= 2 else float("nan")
        ci_half = 1.96 * se_d if n >= 2 else float("nan")
        try:
            _, p = stats.wilcoxon(d, alternative="greater" if mean_d >= 0 else "less")
        except ValueError:
            p = float("nan")
        rows.append(
            {
                "label": label,
                "high": high,
                "low": low,
                "n": n,
                "mean_delta": float(mean_d),
                "se": float(se_d) if not np.isnan(se_d) else None,
                "ci_low": float(mean_d - ci_half) if not np.isnan(ci_half) else None,
                "ci_high": float(mean_d + ci_half) if not np.isnan(ci_half) else None,
                "p": float(p) if not np.isnan(p) else None,
                "significant": bool(((mean_d - ci_half) > 0) or ((mean_d + ci_half) < 0))
                if not np.isnan(ci_half) else False,
            }
        )
    return rows


def compute_subscore_contrast(df, cond_high, cond_low):
    """For each rubric sub-score, compute the H1a-style paired contrast with 95% CI.

    Length-residualizing each sub-score within each prompt before differencing,
    so the test is consistent with the headline H1a treatment.
    """
    score_cols = [
        ("violence_death", "Violence / death"),
        ("bleakness_of_tone", "Bleakness of tone"),
        ("ending_tragic", "Tragic ending"),
        ("character_suffering", "Character suffering"),
        ("overall_darkness", "Overall darkness"),
    ]
    rows = []
    for col, label in score_cols:
        df2 = df.copy()
        df2["resid"] = np.nan
        for pid, sub in df2.groupby("prompt_id"):
            if sub["completion_tokens"].std() < 1 or len(sub) < 3:
                df2.loc[sub.index, "resid"] = sub[col] - sub[col].mean()
                continue
            slope, intercept = np.polyfit(sub["completion_tokens"], sub[col], 1)
            df2.loc[sub.index, "resid"] = sub[col] - (slope * sub["completion_tokens"] + intercept)

        pp = df2.groupby(["prompt_id", "condition"])["resid"].mean().unstack("condition").reset_index()
        if cond_high not in pp.columns or cond_low not in pp.columns:
            continue
        sub = pp.dropna(subset=[cond_high, cond_low])
        d = sub[cond_high].values - sub[cond_low].values
        n = len(d)
        if n < 2:
            continue
        mean_d = d.mean()
        sd_d = d.std(ddof=1)
        se_d = sd_d / np.sqrt(n)
        ci_half = 1.96 * se_d
        try:
            _, p = stats.wilcoxon(d, alternative="greater" if mean_d >= 0 else "less")
        except ValueError:
            p = float("nan")
        rows.append(
            {
                "score": col,
                "label": label,
                "n": n,
                "mean_delta": float(mean_d),
                "ci_low": float(mean_d - ci_half),
                "ci_high": float(mean_d + ci_half),
                "p": float(p) if not np.isnan(p) else None,
                "significant": bool((mean_d - ci_half) > 0 or (mean_d + ci_half) < 0),
            }
        )
    return rows


def compute_paired_scatter(df, cond_high, cond_low):
    """Per-prompt mean darkness in two conditions (raw 0-10 scale)."""
    pp = (
        df.groupby(["prompt_id", "stratum", "fandom", "condition"])["overall_darkness"]
        .mean()
        .reset_index()
    )
    pivot = pp.pivot(
        index=["prompt_id", "stratum", "fandom"],
        columns="condition",
        values="overall_darkness",
    ).reset_index()
    if cond_high not in pivot.columns or cond_low not in pivot.columns:
        return []
    pivot = pivot.dropna(subset=[cond_high, cond_low])
    return [
        {
            "x": float(r[cond_low]),
            "y": float(r[cond_high]),
            "stratum": r["stratum"],
            "fandom": r["fandom"],
            "prompt_id": r["prompt_id"],
        }
        for _, r in pivot.iterrows()
    ]


def pick_examples(valid, per_prompt, n_prompts=8):
    """For top-affected prompts (by H1a delta), pick the most-illustrative
    high-frust and low-frust responses to display side by side.

    Strategy: take the highest-darkness response from possible-high and the
    lowest-darkness response from possible-low for that prompt. Both must
    have non-empty stories.
    """
    examples = []
    top = per_prompt.head(n_prompts)
    for _, row in top.iterrows():
        pid = row["prompt_id"]
        sub = valid[valid["prompt_id"] == pid]
        high_pool = sub[sub["condition"] == "failed-possible-high"]
        low_pool = sub[sub["condition"] == "failed-possible-low"]
        if len(high_pool) == 0 or len(low_pool) == 0:
            continue
        high_pick = high_pool.sort_values("overall_darkness", ascending=False).iloc[0]
        low_pick = low_pool.sort_values("overall_darkness", ascending=True).iloc[0]
        examples.append(
            {
                "prompt_id": pid,
                "stratum": row["stratum"],
                "fandom": row["fandom"],
                "prompt": row["prompt"],
                "delta": row["delta"],
                "mean_high": row["mean_high"],
                "mean_low": row["mean_low"],
                "high": high_pick,
                "low": low_pick,
            }
        )
    return examples


# ---------- HTML rendering ----------

CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    background: #faf8f4;
    color: #2c2520;
    line-height: 1.6;
}
.container { max-width: 1100px; margin: 0 auto; padding: clamp(1.5rem, 4vw, 3rem); }
h1 {
    font-family: 'Newsreader', Georgia, serif;
    font-size: clamp(1.75rem, 3vw, 2.25rem);
    font-weight: 600;
    margin-bottom: 0.5rem;
    letter-spacing: -0.01em;
}
h2 {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 1.35rem;
    font-weight: 500;
    margin: 2.5rem 0 1rem;
    border-bottom: 1px solid #ddd7cd;
    padding-bottom: 0.5rem;
}
h3 {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 1.1rem;
    font-weight: 500;
    margin: 1.5rem 0 0.5rem;
}
h4 {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 0.95rem;
    font-weight: 500;
    margin: 1.5rem 0 0.5rem;
    color: #6b5f54;
}
.subtitle {
    color: #6b5f54;
    margin-bottom: 2rem;
    font-size: 0.95rem;
}
.subtitle strong { color: #2c2520; }
.headline-stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0 2rem;
}
.headline-stat {
    background: #f0ece5;
    border-radius: 8px;
    padding: 1rem 1.2rem;
}
.headline-stat .label {
    color: #6b5f54;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.headline-stat .value {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 1.6rem;
    font-weight: 600;
    margin-top: 0.25rem;
}
.headline-stat .sub { color: #6b5f54; font-size: 0.82rem; margin-top: 0.15rem; }
.chart-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    margin: 1.5rem 0;
}
.chart-box {
    background: #f0ece5;
    border-radius: 8px;
    padding: 1.5rem;
}
.chart-box.full { grid-column: 1 / -1; }
canvas { width: 100% !important; max-height: 400px; }
table {
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
    font-size: 0.9rem;
    font-variant-numeric: tabular-nums;
}
th, td {
    padding: 0.6rem 0.8rem;
    text-align: left;
    border-bottom: 1px solid #ddd7cd;
}
th {
    color: #6b5f54;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
}
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tr:hover { background: #f0ece580; }
.sig { color: #2d7a5f; font-weight: 600; }
.null { color: #9b9088; }
.delta-pos { color: #a63d3d; font-weight: 600; }
.delta-neg { color: #2d7a5f; font-weight: 600; }
.methodology, .paper-ref {
    background: #f0ece5;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.9rem;
    color: #6b5f54;
}
.methodology strong, .paper-ref strong { color: #2c2520; }
.methodology code {
    background: #faf8f4;
    padding: 0.1rem 0.35rem;
    border-radius: 3px;
    font-size: 0.82em;
    border: 1px solid #ddd7cd;
}
.methodology pre {
    background: #faf8f4;
    padding: 0.7rem 0.9rem;
    border-radius: 6px;
    margin-top: 0.4rem;
    font-size: 0.78rem;
    white-space: pre-wrap;
    word-wrap: break-word;
    border: 1px solid #ddd7cd;
    line-height: 1.55;
}
.methodology .label-tag {
    display: block;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #6b5f54;
    margin-top: 0.7rem;
}
.stratum-pill {
    display: inline-block;
    padding: 0.1rem 0.5rem;
    border-radius: 3px;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
    color: white;
}
.stratum-dark { background: #7a1515; }
.stratum-neutral { background: #6b5f54; }
.stratum-light { background: #2d7a5f; }
.example-card {
    margin: 2rem 0 0;
    border-top: 1px solid #eae5db;
    padding-top: 1.5rem;
}
.example-header {
    display: flex;
    align-items: baseline;
    gap: 0.8rem;
    flex-wrap: wrap;
    margin-bottom: 0.5rem;
}
.example-prompt {
    font-style: italic;
    color: #4a4039;
    margin-bottom: 0.5rem;
    font-size: 0.92rem;
}
.example-meta {
    color: #6b5f54;
    font-size: 0.82rem;
    margin-bottom: 1rem;
}
.comparison {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
}
.comparison-col h4 {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 0.95rem;
    font-weight: 500;
    margin-bottom: 0.5rem;
    color: #2c2520;
}
.comparison-label {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 0.3rem;
    display: flex;
    gap: 0.6rem;
    align-items: center;
}
.rating-badge {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-weight: bold;
    font-size: 0.78rem;
    color: white;
}
.judge-evidence {
    color: #8a6d3b;
    font-style: italic;
    margin: 0.4rem 0 0.5rem;
    font-size: 0.85rem;
}
.story {
    font-size: 0.85rem;
    color: #4a4039;
    line-height: 1.6;
    max-height: 360px;
    overflow-y: auto;
    background: #f5f1ea;
    border-radius: 6px;
    border: 1px solid #eae5db;
    padding: 0.9rem 1rem;
    white-space: pre-wrap;
    word-wrap: break-word;
}
.story::-webkit-scrollbar { width: 4px; }
.story::-webkit-scrollbar-thumb { background: #ddd7cd; border-radius: 2px; }
.note {
    color: #6b5f54;
    font-size: 0.85rem;
    margin: 0.5rem 0;
}
@media (max-width: 800px) {
    .chart-row { grid-template-columns: 1fr; }
    .comparison { grid-template-columns: 1fr; }
}
"""


def darkness_color(score):
    if score < 2:
        return "#2d7a5f"
    if score < 4:
        return "#7a9bcc"
    if score < 6:
        return "#c5a574"
    if score < 8:
        return "#a63d3d"
    return "#7a1515"


def fmt_p(p):
    if pd.isna(p):
        return "—"
    if p < 0.001:
        return f"<span class='sig'>p&lt;0.001</span>"
    if p < 0.05:
        return f"<span class='sig'>p={p:.3f}</span>"
    return f"<span class='null'>p={p:.3f}</span>"


def fmt_delta(d, signed=True):
    cls = "delta-pos" if d > 0 else "delta-neg"
    sign = "+" if (signed and d >= 0) else ""
    return f"<span class='{cls}'>{sign}{d:.3f}</span>"


def render_condition_table(stats_df):
    rows = ""
    for _, r in stats_df.iterrows():
        rows += (
            f"<tr><td>{escape(r['condition'])}</td>"
            f"<td class='num'>{int(r['n'])}</td>"
            f"<td class='num'>{r['mean']:.3f}</td>"
            f"<td class='num'>±{r['se']:.3f}</td>"
            f"<td class='num'>{r['violence']:.2f}</td>"
            f"<td class='num'>{r['bleakness']:.2f}</td>"
            f"<td class='num'>{r['tragic']:.2f}</td>"
            f"<td class='num'>{r['suffering']:.2f}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Condition</th><th class='num'>n</th>"
        "<th class='num'>Overall</th><th class='num'>SE</th>"
        "<th class='num'>Violence</th><th class='num'>Bleakness</th>"
        "<th class='num'>Tragic</th><th class='num'>Suffering</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def render_stratum_table(stratum_df):
    pivot = stratum_df.pivot(index="condition", columns="stratum", values="mean")
    pivot = pivot.reindex(CONDITIONS)
    rows = ""
    for cond in pivot.index:
        if pivot.loc[cond].isna().all():
            continue
        cells = ""
        for s in ["dark", "neutral", "light"]:
            v = pivot.loc[cond].get(s)
            if pd.isna(v):
                cells += "<td class='num'>—</td>"
            else:
                cells += f"<td class='num' style='color:{darkness_color(v)};font-weight:600'>{v:.2f}</td>"
        rows += f"<tr><td>{escape(cond)}</td>{cells}</tr>"
    return (
        "<table><thead><tr><th>Condition</th>"
        "<th class='num'>Dark</th><th class='num'>Neutral</th><th class='num'>Light</th>"
        "</tr></thead><tbody>" + rows + "</tbody></table>"
    )


def render_per_prompt_table(per_prompt):
    rows = ""
    for _, r in per_prompt.iterrows():
        stratum_class = f"stratum-{r['stratum']}"
        rows += (
            f"<tr>"
            f"<td><span class='stratum-pill {stratum_class}'>{r['stratum']}</span></td>"
            f"<td>{escape(r['fandom'])}</td>"
            f"<td>{escape(r['prompt'])}</td>"
            f"<td class='num'>{r['mean_low']:.2f}</td>"
            f"<td class='num'>{r['mean_high']:.2f}</td>"
            f"<td class='num'>{fmt_delta(r['delta'])}</td>"
            f"</tr>"
        )
    return (
        "<table><thead><tr><th>Stratum</th><th>Fandom</th><th>Prompt</th>"
        "<th class='num'>Low</th><th class='num'>High</th><th class='num'>Δ</th>"
        "</tr></thead><tbody>" + rows + "</tbody></table>"
    )


def render_stratum_delta_table(stratum_delta):
    rows = ""
    for _, r in stratum_delta.iterrows():
        rows += (
            f"<tr>"
            f"<td><span class='stratum-pill stratum-{r['stratum']}'>{r['stratum']}</span></td>"
            f"<td class='num'>{int(r['n'])}</td>"
            f"<td class='num'>{fmt_delta(r['mean_delta'])}</td>"
            f"<td class='num'>{r['cohens_d']:+.3f}</td>"
            f"<td class='num'>{fmt_p(r['p'])}</td>"
            f"</tr>"
        )
    return (
        "<table><thead><tr><th>Stratum</th><th class='num'>n prompts</th>"
        "<th class='num'>Mean Δ</th><th class='num'>Cohen's d</th>"
        "<th class='num'>Wilcoxon p (1-sided)</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def render_length_table(length_contrasts):
    rows = ""
    labels = {
        "H1a": "H1a — possible-high vs possible-low",
        "H1b": "H1b — impossible-high vs impossible-low",
        "ph_vs_success": "possible-high vs success",
    }
    for k, v in length_contrasts.items():
        rows += (
            f"<tr><td>{escape(labels[k])}</td>"
            f"<td class='num'>{int(v['n'])}</td>"
            f"<td class='num'>{fmt_delta(v['mean_delta'])}</td>"
            f"<td class='num'>{v['cohens_d']:+.3f}</td>"
            f"<td class='num'>{fmt_p(v['p'])}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Contrast</th><th class='num'>n prompts</th>"
        "<th class='num'>Mean Δ (residualized)</th><th class='num'>Cohen's d</th>"
        "<th class='num'>Wilcoxon p</th></tr></thead><tbody>" + rows + "</tbody></table>"
    )


def render_examples(examples):
    cards = ""
    for ex in examples:
        high_dark = ex["high"]["overall_darkness"]
        low_dark = ex["low"]["overall_darkness"]
        cards += f"""
        <div class="example-card">
            <div class="example-header">
                <span class="stratum-pill stratum-{ex['stratum']}">{ex['stratum']}</span>
                <strong>{escape(ex['fandom'])}</strong>
                <span class='example-meta'>per-prompt Δ = {ex['delta']:+.2f}
                (high mean {ex['mean_high']:.2f}, low mean {ex['mean_low']:.2f})</span>
            </div>
            <div class="example-prompt">"{escape(ex['prompt'])}"</div>
            <div class="comparison">
                <div class="comparison-col">
                    <div class="comparison-label" style="color:#7a9bcc">
                        failed-possible-low
                        <span class="rating-badge" style="background:{darkness_color(low_dark)}">{low_dark:.0f}/10</span>
                    </div>
                    <div class="judge-evidence">"{escape(str(ex['low'].get('evidence', ''))[:200])}"</div>
                    <div class="story">{escape(str(ex['low'].get('response', '')))}</div>
                </div>
                <div class="comparison-col">
                    <div class="comparison-label" style="color:#a63d3d">
                        failed-possible-high
                        <span class="rating-badge" style="background:{darkness_color(high_dark)}">{high_dark:.0f}/10</span>
                    </div>
                    <div class="judge-evidence">"{escape(str(ex['high'].get('evidence', ''))[:200])}"</div>
                    <div class="story">{escape(str(ex['high'].get('response', '')))}</div>
                </div>
            </div>
        </div>
        """
    return cards


def render_html(
    stats_df,
    stratum_df,
    per_prompt,
    stratum_delta,
    length_contrasts,
    length_mdf,
    examples,
    contrast_estimates,
    paired_scatter,
    subscore_contrasts,
    n_total,
    n_valid,
    refusal_total,
    n_failed_judge,
):
    headline_d = length_contrasts["H1a"]["cohens_d"]
    headline_delta = length_contrasts["H1a"]["mean_delta"]
    headline_p = length_contrasts["H1a"]["p"]

    cond_data = [
        {
            "label": r["condition"],
            "y": float(r["mean"]),
            "yMin": float(r["mean"] - 1.96 * r["se"]),
            "yMax": float(r["mean"] + 1.96 * r["se"]),
            "n": int(r["n"]),
        }
        for _, r in stats_df.iterrows()
    ]

    pivot_mean = stratum_df.pivot(index="condition", columns="stratum", values="mean").reindex(CONDITIONS)
    pivot_se = stratum_df.pivot(index="condition", columns="stratum", values="se").reindex(CONDITIONS)
    stratum_data = []
    for s in ["dark", "neutral", "light"]:
        cells = []
        for c in CONDITIONS:
            m = pivot_mean.loc[c].get(s)
            se = pivot_se.loc[c].get(s)
            if pd.isna(m):
                cells.append({"y": None, "yMin": None, "yMax": None})
            else:
                cells.append(
                    {
                        "y": float(m),
                        "yMin": float(m - 1.96 * se),
                        "yMax": float(m + 1.96 * se),
                    }
                )
        stratum_data.append({"stratum": s, "cells": cells})

    delta_data = [
        {
            "prompt_id": r["prompt_id"],
            "fandom": r["fandom"],
            "stratum": r["stratum"],
            "x": float(r["delta"]),
            "xMin": float(r["ci_low"]),
            "xMax": float(r["ci_high"]),
        }
        for _, r in per_prompt.iterrows()
    ]

    cond_table_html = render_condition_table(stats_df)
    stratum_table_html = render_stratum_table(stratum_df)
    full_per_prompt_html = render_per_prompt_table(per_prompt)
    top_per_prompt_html = render_per_prompt_table(per_prompt.head(8))
    bot_per_prompt_html = render_per_prompt_table(per_prompt.tail(8))
    stratum_delta_html = render_stratum_delta_table(stratum_delta)
    length_table_html = render_length_table(length_contrasts)
    examples_html = render_examples(examples)

    cond_data_json = json.dumps(cond_data)
    stratum_data_json = json.dumps(stratum_data)
    delta_data_json = json.dumps(delta_data)
    conditions_json = json.dumps(CONDITIONS)
    stratum_colors_json = json.dumps(STRATUM_COLORS)
    cond_colors_json = json.dumps([CONDITION_COLORS[c] for c in CONDITIONS])
    contrast_data_json = json.dumps(contrast_estimates)
    scatter_data_json = json.dumps(paired_scatter)
    subscore_data_json = json.dumps(subscore_contrasts)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Does Frustrated Gemma Write Darker Fanfiction?</title>
<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=DM+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-chart-error-bars@4"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3"></script>
<style>{CSS}</style>
</head>
<body>
<div class="container">

<h1>Does Frustrated Gemma Write Darker Fanfiction?</h1>
<p class="subtitle">
    A pre-registered test of whether Gemma-3-27B-it's emotional state — induced by failure on
    arithmetic puzzles — spills into the tone of unrelated short fanfiction prompts.
    <strong>Headline:</strong> yes, but only when the puzzle failure is genuine.
    Frustration on a solvable-but-failed puzzle shifts mean fanfic darkness by
    <strong>{headline_delta:+.3f}</strong> on a 0–10 scale (Cohen's d = <strong>{headline_d:+.2f}</strong>,
    one-sided Wilcoxon {fmt_p(headline_p).replace("<span class='sig'>", "").replace("</span>", "").replace("&lt;", "<")}).
    Frustration on the gaslit-impossible puzzle does not produce the effect.
    The shift survives controlling for story length.
</p>

<div class="headline-stats">
    <div class="headline-stat">
        <div class="label">H1a (possible)</div>
        <div class="value sig">supported</div>
        <div class="sub">Δ={length_contrasts['H1a']['mean_delta']:+.3f}, d={length_contrasts['H1a']['cohens_d']:+.2f}</div>
    </div>
    <div class="headline-stat">
        <div class="label">H1b (impossible)</div>
        <div class="value null">null</div>
        <div class="sub">Δ={length_contrasts['H1b']['mean_delta']:+.3f}, d={length_contrasts['H1b']['cohens_d']:+.2f}</div>
    </div>
    <div class="headline-stat">
        <div class="label">Refusals</div>
        <div class="value">{refusal_total} / {n_total}</div>
        <div class="sub">across all conditions</div>
    </div>
    <div class="headline-stat">
        <div class="label">H1a effect size</div>
        <div class="value">d = {length_contrasts['H1a']['cohens_d']:+.2f}</div>
        <div class="sub">Δ = {length_contrasts['H1a']['mean_delta']:+.3f} pt on 0–10 scale</div>
    </div>
</div>

<h2>Effect at a glance</h2>
<p class="note">
    The right way to read a paired hypothesis test visually: <strong>plot the contrast (the difference)
    with its 95% CI, not the two means side-by-side</strong>. The per-condition mean bars further down
    have CIs of ±0.17 because they include between-prompt variance (some prompts are naturally dark,
    some light). But the test is <em>paired within prompt</em> — the between-prompt component cancels.
    The forest plot below shows the same data on the difference scale, where the relevant 95% CI is
    ~5× tighter. <strong>A CI that excludes zero corresponds to a significant directional test.</strong>
</p>
<div class="chart-row">
    <div class="chart-box"><canvas id="contrastChart"></canvas></div>
    <div class="chart-box"><canvas id="scatterChart"></canvas></div>
</div>
<p class="note">
    <strong>Left:</strong> mean of per-prompt deltas with 95% CI (length-residualized darkness).
    <span style="color:#a63d3d;font-weight:600">Red</span> = CI excludes zero (significant);
    <span style="color:#9b9088;font-weight:600">grey</span> = CI crosses zero (not significant).
    The H1a contrast is the headline result — its CI clearly sits above zero.
    <strong>Right:</strong> each dot is one of 40 prompts. X = mean darkness under failed-possible-low,
    Y = mean darkness under failed-possible-high. The dashed diagonal is y = x. Dots above the
    diagonal got darker under high frustration. The cluster's overall offset above the diagonal
    is the H1a effect, eyeballable.
</p>

<h2>Methodology</h2>
<div class="methodology">
    <p>
        Adapted from <a href="https://www.lesswrong.com/posts/kjnQj6YujgeMN9Erq/gemma-needs-help">Gemma Needs Help</a>
        (Soligo et al., 2026). Gemma-3-27B-it is given a multi-turn arithmetic puzzle conversation as its
        history (the <em>prefix</em>), then asked to write a short fanfiction scene. The fanfic response is
        scored for darkness by a blind judge that sees only the prompt and the story.
    </p>
    <span class="label-tag">Conditions (6)</span>
    <ul style="margin: 0.4rem 0 0 1.2rem; font-size: 0.88rem;">
        <li><code>blank</code> — no prefix.</li>
        <li><code>success</code> — Gemma solved hard solvable puzzles, got positive feedback.</li>
        <li><code>failed-possible-low</code> — Gemma failed solvable puzzles, judged frustration &lt; 3.</li>
        <li><code>failed-possible-high</code> — Gemma failed solvable puzzles, judged frustration ≥ 5.</li>
        <li><code>failed-impossible-low</code> — Gemma failed the gaslit-impossible puzzle, frustration &lt; 3.</li>
        <li><code>failed-impossible-high</code> — Gemma failed the gaslit-impossible puzzle, frustration ≥ 5.</li>
    </ul>
    <span class="label-tag">Prompts</span>
    <p>40 fanfiction premises, all alternate-universe (no canon-extension, no crossovers, no role-swaps),
    stratified into 16 dark, 14 neutral, and 10 light baseline strata. Each is a <em>premise</em>, not a <em>plot</em> —
    the model picks what happens within the AU.</p>
    <span class="label-tag">Generations</span>
    <p>For each (prompt × condition × prefix instance), 30 independent runs at temperature 1.0
    (no <code>seed</code> parameter — provider support is unreliable; variance comes from sampling).
    Up to 3 prefix instances per condition. Total: <strong>{n_total:,} generations</strong>.</p>
    <span class="label-tag">Outcome measure</span>
    <p>Claude Haiku 4.5 (temperature = 0) reads each story and returns a 0–10 darkness score
    (plus four sub-scores: violence, bleakness, tragic-ending, suffering). The judge also flags
    refusals (any response that is not a fanfic story); refusals are excluded from the darkness
    analysis. The judge sees only the fanfic prompt and the story — never the prefix or condition label.</p>
    <span class="label-tag">Exact judge rubric (system prompt)</span>
    <pre>{escape(JUDGE_SYSTEM_PROMPT.strip())}</pre>
</div>

<h2>Mean darkness by condition (between-prompt view)</h2>
<p class="note">
    Reference view of the per-condition means with 95% CIs that pool over all 40 prompts. The CIs
    look wide because they include the large between-prompt variance (dark prompts ≈ 7/10, light ≈ 0/10).
    These bars are NOT how you should read the hypothesis test — the paired structure is invisible here.
    See the forest plot above for the correct visualization of the contrast.
</p>
<div class="chart-row">
    <div class="chart-box"><canvas id="condChart"></canvas></div>
    <div class="chart-box"><canvas id="stratumChart"></canvas></div>
</div>

{cond_table_html}

<h3>Stratified by prompt baseline darkness</h3>
{stratum_table_html}

<h2>The primary contrasts</h2>
<p class="note">
    The primary tests are within-prompt paired Wilcoxon signed-rank, one-sided greater.
    Each prompt contributes one delta = mean(high-frust condition) − mean(low-frust condition).
</p>
{length_table_html}
<p class="note">
    Reported on <em>length-residualized</em> darkness — story length is regressed out within each
    prompt before averaging. Raw and residualized results agree closely (residualization slightly
    strengthens the H1a effect, since within-prompt variation in length is mildly negatively
    correlated with darkness once condition is held fixed).
</p>

<h2>Per-stratum effect</h2>
<p class="note">Does the H1a effect (failed-possible-high vs failed-possible-low) depend on the prompt's
baseline darkness stratum?</p>
{stratum_delta_html}

<h2>Which sub-dimension drives the effect?</h2>
<p class="note">
    The judge rubric scores 4 sub-dimensions plus an overall holistic darkness. The forest plot below
    runs the same H1a contrast (failed-possible-high − failed-possible-low, length-residualized) on
    each sub-score separately. Sub-scores whose CI excludes zero are highlighted in red.
</p>
<div class="chart-box full" style="margin: 1rem 0;">
    <canvas id="subscoreChart"></canvas>
</div>
<p class="note">
    <strong>Reading:</strong> the H1a effect is concentrated in <em>tragic endings</em> (largest shift, CI
    cleanly above zero). Bleakness of tone is borderline. Violence/death and character suffering do
    NOT shift — frustrated Gemma is not writing more gore or more cruelty, it's writing endings with
    more grief, loss, or doom. This is consistent with the original n=1 GoT/Red-Wedding observation,
    which was specifically described as "a more tragic ending."
</p>

<h2>Per-prompt deltas</h2>
<p class="note">
    Sorted by mean(failed-possible-high) − mean(failed-possible-low). Top 8 are above the gap; bottom 8 below.
    Click the chart for the full picture across 40 prompts.
</p>
<div class="chart-box full" style="margin: 1.5rem 0;">
    <canvas id="deltaChart"></canvas>
</div>

<h3>Top 8 prompts (largest positive shift = darker under high frustration)</h3>
{top_per_prompt_html}

<h3>Bottom 8 prompts</h3>
{bot_per_prompt_html}

<details style="margin-top: 1rem;">
    <summary style="cursor:pointer;color:#3a6fa5">All 40 prompts</summary>
    {full_per_prompt_html}
</details>

<h2>Example response pairs</h2>
<p class="note">
    For each of the top-8 most-affected prompts (by H1a delta), we show one
    failed-possible-high response (highest darkness in that prompt's pool) alongside one
    failed-possible-low response (lowest darkness). Same prompt, different prefix condition.
</p>
{examples_html}

<h2>Length confound check (PREREG-mandated)</h2>
<p class="note">
    The PREREG required a length-covariate check because <code>failed-possible-high</code>
    produced ~7% longer stories than <code>failed-possible-low</code>. The mixed-effects model
    below adds <code>completion_tokens</code> as a fixed effect (centered) and prompt as random intercept.
    The condition coefficient survives — frustration is not just verbosity in disguise.
</p>
<pre style="background:#f0ece5;border-radius:8px;padding:1rem 1.2rem;font-size:0.78rem;overflow-x:auto;line-height:1.4">{escape(length_mdf.summary().as_text())}</pre>

<h2>Sanity checks</h2>
<table>
    <tbody>
    <tr><td>Total generations</td><td class='num'>{n_total}</td></tr>
    <tr><td>Generation failures (None responses)</td><td class='num'>{n_total - n_valid - refusal_total - n_failed_judge:+d} (placeholder)</td></tr>
    <tr><td>Refusals (judge-flagged non-stories)</td><td class='num'>{refusal_total}</td></tr>
    <tr><td>Failed judge parses</td><td class='num'>{n_failed_judge}</td></tr>
    <tr><td>Valid scoring records</td><td class='num'>{n_valid}</td></tr>
    </tbody>
</table>

<h2>Limitations</h2>
<ul style="margin-left:1.2rem;font-size:0.92rem;color:#4a4039;">
    <li><strong>Effect size is small.</strong> {length_contrasts['H1a']['mean_delta']:+.3f} points on a 0–10 scale, d = {length_contrasts['H1a']['cohens_d']:+.2f}. Reliable on the paired test, but not dramatic — a single story under either condition would not look noticeably different from one under the other; the effect only emerges as a systematic shift averaged across the 40 prompts.</li>
    <li><strong>Single judge.</strong> Claude Haiku 4.5 only, k=1 pass per story. No inter-rater reliability check (the PREREG flagged this as an optional follow-up).</li>
    <li><strong>One model.</strong> Gemma-3-27B-it only. The original "Gemma Needs Help" paper showed other model families exhibit minimal frustration in the first place, so this experiment likely doesn't generalize beyond Gemma.</li>
    <li><strong>One prefix family.</strong> Frustration was always induced via arithmetic puzzles. Real-world frustration could come from coding, writing, or roleplay tasks and might not transfer the same way.</li>
    <li><strong>The H1b null is informative but not crisp.</strong> The impossible-puzzle prefixes were rated as more frustrated by the judge but did not shift fanfic darkness. The cleanest reading is that the gaslit-puzzle frustration may be qualitatively different from genuine task-failure frustration, but we cannot rule out that the impossible-puzzle pool happens to contain noisier or otherwise atypical prefixes.</li>
</ul>

</div>

<script>
const conditions = {conditions_json};
const condColors = {cond_colors_json};
const condData = {cond_data_json};
const stratumData = {stratum_data_json};
const deltaData = {delta_data_json};
const stratumColors = {stratum_colors_json};
const contrastData = {contrast_data_json};
const scatterData = {scatter_data_json};

const fontFamily = "'DM Sans', sans-serif";
Chart.defaults.font.family = fontFamily;
Chart.defaults.color = '#4a4039';

// ---- Contrast forest plot (paired CIs, the right viz for the test) ----
new Chart(document.getElementById('contrastChart'), {{
    type: 'barWithErrorBars',
    data: {{
        labels: contrastData.map(c => c.label),
        datasets: [{{
            label: 'Δ darkness (length-residualized, 95% CI)',
            data: contrastData.map(c => ({{
                x: c.mean_delta,
                xMin: c.ci_low,
                xMax: c.ci_high,
            }})),
            backgroundColor: contrastData.map(c => c.significant ? '#a63d3d' : '#9b9088'),
            borderRadius: 3,
            errorBarLineWidth: 1.5,
            errorBarColor: '#2c2520',
            errorBarWhiskerLineWidth: 1.5,
            errorBarWhiskerSize: 8,
        }}],
    }},
    options: {{
        indexAxis: 'y',
        responsive: true,
        plugins: {{
            legend: {{ display: false }},
            title: {{ display: true, text: 'Contrast estimates with 95% CI (paired across 40 prompts)', font: {{ size: 14 }} }},
            annotation: {{
                annotations: {{
                    zeroLine: {{
                        type: 'line',
                        scaleID: 'x',
                        value: 0,
                        borderColor: '#2c2520',
                        borderWidth: 2,
                        borderDash: [4, 4],
                        label: {{ display: false }},
                    }}
                }}
            }},
            tooltip: {{
                callbacks: {{
                    label: (ctx) => {{
                        const c = contrastData[ctx.dataIndex];
                        const pStr = (c.p == null) ? 'n/a' : c.p.toFixed(4);
                        return `Δ=${{c.mean_delta.toFixed(3)}}  95% CI [${{c.ci_low.toFixed(3)}}, ${{c.ci_high.toFixed(3)}}]  p=${{pStr}}  n=${{c.n}}`;
                    }}
                }}
            }}
        }},
        scales: {{
            x: {{ title: {{ display: true, text: 'Δ darkness (length-residualized)' }}, grid: {{ color: '#ddd7cd' }} }},
            y: {{ ticks: {{ font: {{ size: 11 }}, autoSkip: false }} }}
        }}
    }}
}});

// ---- Per-sub-score forest plot ----
const subscoreData = {subscore_data_json};
new Chart(document.getElementById('subscoreChart'), {{
    type: 'barWithErrorBars',
    data: {{
        labels: subscoreData.map(s => s.label),
        datasets: [{{
            label: 'H1a Δ on sub-score (length-residualized, 95% CI)',
            data: subscoreData.map(s => ({{
                x: s.mean_delta,
                xMin: s.ci_low,
                xMax: s.ci_high,
            }})),
            backgroundColor: subscoreData.map(s => s.significant ? '#a63d3d' : '#9b9088'),
            borderRadius: 3,
            errorBarLineWidth: 1.5,
            errorBarColor: '#2c2520',
            errorBarWhiskerLineWidth: 1.5,
            errorBarWhiskerSize: 8,
        }}],
    }},
    options: {{
        indexAxis: 'y',
        responsive: true,
        plugins: {{
            legend: {{ display: false }},
            title: {{ display: true, text: 'H1a contrast by sub-score (paired across 40 prompts)', font: {{ size: 14 }} }},
            annotation: {{
                annotations: {{
                    zeroLine: {{
                        type: 'line',
                        scaleID: 'x',
                        value: 0,
                        borderColor: '#2c2520',
                        borderWidth: 2,
                        borderDash: [4, 4],
                        label: {{ display: false }},
                    }}
                }}
            }},
            tooltip: {{
                callbacks: {{
                    label: (ctx) => {{
                        const s = subscoreData[ctx.dataIndex];
                        const pStr = (s.p == null) ? 'n/a' : s.p.toFixed(4);
                        return `Δ=${{s.mean_delta.toFixed(3)}}  95% CI [${{s.ci_low.toFixed(3)}}, ${{s.ci_high.toFixed(3)}}]  p=${{pStr}}`;
                    }}
                }}
            }}
        }},
        scales: {{
            x: {{ title: {{ display: true, text: 'Δ on sub-score (0–10 scale, length-residualized)' }}, grid: {{ color: '#ddd7cd' }} }},
            y: {{ ticks: {{ font: {{ size: 11 }}, autoSkip: false }} }}
        }}
    }}
}});

// ---- Paired scatter (each dot = one prompt) ----
new Chart(document.getElementById('scatterChart'), {{
    type: 'scatter',
    data: {{
        datasets: ['dark', 'neutral', 'light'].map(s => ({{
            label: s,
            data: scatterData.filter(d => d.stratum === s),
            backgroundColor: stratumColors[s],
            borderColor: stratumColors[s],
            pointRadius: 5,
            pointHoverRadius: 7,
        }})),
    }},
    options: {{
        responsive: true,
        aspectRatio: 1,
        plugins: {{
            legend: {{ display: true, position: 'top', labels: {{ font: {{ size: 11 }}, boxWidth: 12 }} }},
            title: {{ display: true, text: 'Per-prompt: low-frust (x) vs high-frust (y), above diagonal = darker under high frust', font: {{ size: 12 }} }},
            annotation: {{
                annotations: {{
                    diagonal: {{
                        type: 'line',
                        xMin: 0, xMax: 10,
                        yMin: 0, yMax: 10,
                        borderColor: '#9b9088',
                        borderWidth: 1.5,
                        borderDash: [6, 6],
                        label: {{ display: false }},
                    }}
                }}
            }},
            tooltip: {{
                callbacks: {{
                    label: (ctx) => {{
                        const d = ctx.raw;
                        return `${{d.fandom}} (${{d.stratum}}): low=${{d.x.toFixed(2)}}, high=${{d.y.toFixed(2)}}`;
                    }}
                }}
            }}
        }},
        scales: {{
            x: {{ min: 0, max: 10, title: {{ display: true, text: 'Mean darkness  —  failed-possible-low' }}, grid: {{ color: '#ddd7cd' }} }},
            y: {{ min: 0, max: 10, title: {{ display: true, text: 'Mean darkness  —  failed-possible-high' }}, grid: {{ color: '#ddd7cd' }} }}
        }}
    }}
}});

new Chart(document.getElementById('condChart'), {{
    type: 'barWithErrorBars',
    data: {{
        labels: conditions,
        datasets: [{{
            label: 'Mean overall darkness (95% CI)',
            data: condData,
            backgroundColor: condColors,
            borderColor: '#2c2520',
            borderWidth: 0,
            borderRadius: 4,
            errorBarLineWidth: 1.5,
            errorBarColor: '#2c2520',
            errorBarWhiskerLineWidth: 1.5,
            errorBarWhiskerSize: 8,
        }}],
    }},
    options: {{
        responsive: true,
        plugins: {{
            legend: {{ display: false }},
            title: {{ display: true, text: 'Mean overall darkness by condition (95% CI)', font: {{ size: 14 }} }},
            tooltip: {{
                callbacks: {{
                    afterLabel: (ctx) => {{
                        const d = condData[ctx.dataIndex];
                        return `n=${{d.n}}\\n95% CI [${{d.yMin.toFixed(3)}}, ${{d.yMax.toFixed(3)}}]`;
                    }}
                }}
            }}
        }},
        scales: {{
            y: {{ beginAtZero: false, suggestedMin: 3.4, suggestedMax: 3.9, title: {{ display: true, text: 'Darkness' }} }},
            x: {{ ticks: {{ font: {{ size: 10 }} }} }}
        }}
    }}
}});

new Chart(document.getElementById('stratumChart'), {{
    type: 'barWithErrorBars',
    data: {{
        labels: conditions,
        datasets: stratumData.map(s => ({{
            label: s.stratum,
            data: s.cells,
            backgroundColor: stratumColors[s.stratum],
            borderRadius: 4,
            errorBarLineWidth: 1.2,
            errorBarColor: '#2c2520',
            errorBarWhiskerLineWidth: 1.2,
            errorBarWhiskerSize: 6,
        }})),
    }},
    options: {{
        responsive: true,
        plugins: {{
            title: {{ display: true, text: 'Mean darkness by condition × stratum (95% CI)', font: {{ size: 14 }} }}
        }},
        scales: {{
            y: {{ beginAtZero: true, title: {{ display: true, text: 'Darkness' }} }},
            x: {{ ticks: {{ font: {{ size: 10 }} }} }}
        }}
    }}
}});

new Chart(document.getElementById('deltaChart'), {{
    type: 'barWithErrorBars',
    data: {{
        labels: deltaData.map(d => d.fandom),
        datasets: [{{
            label: 'Δ darkness (high − low, 95% CI)',
            data: deltaData,
            backgroundColor: deltaData.map(d => stratumColors[d.stratum]),
            borderRadius: 3,
            errorBarLineWidth: 1,
            errorBarColor: '#2c2520',
            errorBarWhiskerLineWidth: 1,
            errorBarWhiskerSize: 6,
        }}],
    }},
    options: {{
        indexAxis: 'y',
        responsive: true,
        plugins: {{
            legend: {{ display: false }},
            title: {{ display: true, text: 'Per-prompt H1a delta (failed-possible-high − failed-possible-low), 95% CI, sorted', font: {{ size: 14 }} }},
            tooltip: {{
                callbacks: {{
                    title: (items) => items.map(i => deltaData[i.dataIndex].prompt_id),
                    label: (ctx) => {{
                        const d = deltaData[ctx.dataIndex];
                        return `${{d.fandom}} (${{d.stratum}}): Δ=${{d.x.toFixed(3)}} [${{d.xMin.toFixed(3)}}, ${{d.xMax.toFixed(3)}}]`;
                    }}
                }}
            }}
        }},
        scales: {{
            x: {{ title: {{ display: true, text: 'Δ darkness' }}, grid: {{ color: '#ddd7cd' }} }},
            y: {{ ticks: {{ font: {{ size: 9 }}, autoSkip: false }} }}
        }}
    }}
}});
</script>

</body>
</html>
"""


@click.command()
@click.option(
    "--judgments-file",
    default=str(DATA_DIR / "judgments" / "20260423_004557.jsonl"),
)
@click.option(
    "--responses-file",
    default=str(DATA_DIR / "responses" / "20260423_002731.jsonl"),
)
@click.option("--out", default=str(PROJECT_ROOT / "report.html"))
def main(judgments_file, responses_file, out):
    console.print(f"Loading data...")
    merged, valid = load_data(judgments_file, responses_file)
    n_total = len(merged)
    n_valid = len(valid)

    refusal_total = 0
    if "is_refusal" in merged.columns:
        refusal_total = int(merged["is_refusal"].fillna(False).astype(bool).sum())
    n_failed_judge = int((merged["overall_darkness"] < 0).sum()) - refusal_total

    console.print(f"Computing stats...")
    cond_stats = compute_condition_stats(valid)
    stratum_stats = compute_stratum_stats(valid)
    per_prompt = compute_per_prompt_delta(valid, "failed-possible-high", "failed-possible-low")
    stratum_delta = stratum_delta_test(per_prompt)
    length_mdf, length_contrasts = length_covariate_analysis(valid)
    contrast_estimates = compute_contrast_estimates(valid)
    paired_scatter = compute_paired_scatter(valid, "failed-possible-high", "failed-possible-low")
    subscore_contrasts = compute_subscore_contrast(valid, "failed-possible-high", "failed-possible-low")

    console.print(f"Picking example response pairs...")
    examples = pick_examples(valid, per_prompt, n_prompts=8)

    console.print(f"Rendering HTML...")
    html_text = render_html(
        cond_stats,
        stratum_stats,
        per_prompt,
        stratum_delta,
        length_contrasts,
        length_mdf,
        examples,
        contrast_estimates,
        paired_scatter,
        subscore_contrasts,
        n_total=n_total,
        n_valid=n_valid,
        refusal_total=refusal_total,
        n_failed_judge=n_failed_judge,
    )

    Path(out).write_text(html_text)
    console.print(f"[green]Wrote {out} ({len(html_text):,} bytes)[/green]")


if __name__ == "__main__":
    main()
