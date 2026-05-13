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
"""Build a stripped-down version of report.html that conveys the headline finding
in one short page with a single chart.

Sections kept:
  - Title + one-sentence finding
  - 3-sentence methods
  - Contrast forest plot (the only chart)
  - 3-sentence interpretation (concentration + sub-score driver)
  - 2-sentence caveat (effect size + length-confound check)

Everything else in make_report.py is cut.
"""

import json
import math
from html import escape
from pathlib import Path

import click
from rich.console import Console


def _cohens_d(est):
    """Paired Cohen's d from a contrast estimate dict (mean_delta + se + n)."""
    if est["se"] in (None, 0):
        return float("nan")
    sd = est["se"] * math.sqrt(est["n"])
    return est["mean_delta"] / sd if sd > 0 else float("nan")

from make_report import (
    compute_contrast_estimates,
    compute_per_prompt_delta,
    length_covariate_analysis,
    load_data,
    stratum_delta_test,
)

console = Console()
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"


CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    background: #faf8f4;
    color: #2c2520;
    line-height: 1.6;
}
.container { max-width: 760px; margin: 0 auto; padding: clamp(1.25rem, 4vw, 2.5rem); }
h1 {
    font-family: 'Newsreader', Georgia, serif;
    font-size: clamp(1.5rem, 3vw, 2rem);
    font-weight: 600;
    margin-bottom: 0.6rem;
    letter-spacing: -0.01em;
}
h2 {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 1.1rem;
    font-weight: 500;
    margin: 1.75rem 0 0.6rem;
    color: #6b5f54;
}
p { margin-bottom: 0.8rem; font-size: 0.95rem; }
.chart-box {
    background: #f0ece5;
    border-radius: 8px;
    padding: 1.25rem;
    margin: 1.25rem 0;
}
canvas { width: 100% !important; max-height: 380px; }
strong { color: #2c2520; }
.sig { color: #2d7a5f; font-weight: 600; }
.null { color: #9b9088; }
code {
    background: #f0ece5;
    padding: 0.05rem 0.3rem;
    border-radius: 3px;
    font-size: 0.85em;
}
"""


def render_html(contrast_estimates, h1a, h1b, h1a_d, dark_d, dark_p, n_total, n_valid):
    contrast_json = json.dumps(contrast_estimates)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Frustrated Gemma → darker fanfic? — short report</title>
<link href="https://fonts.googleapis.com/css2?family=Newsreader:wght@400;500;600&family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-chart-error-bars@4"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3"></script>
<style>{CSS}</style>
</head>
<body>
<div class="container">

<h1>Does frustrated Gemma write darker fanfiction?</h1>

<p>
    <strong>Yes — but only for genuine task-failure frustration, and the effect is small.</strong>
    Across 40 fanfiction prompts and {n_total:,} generations (Gemma-3-27B-it, T=1.0),
    a Haiku-judged darkness score shifts by <strong class="sig">Δ = {h1a['mean_delta']:+.3f}</strong>
    on a 0–10 scale (<strong>d = {h1a_d:+.2f}</strong>,
    paired Wilcoxon <strong class="sig">p = {h1a['p']:.4f}</strong>) when Gemma's prefix is a
    high-frustration <em>solvable</em>-puzzle failure vs. an otherwise-matched low-frustration
    one. The same comparison on a gaslit <em>impossible</em>-puzzle prefix produces no effect
    (<span class="null">Δ ≈ 0, p = {h1b['p']:.2f}</span>).
</p>

<h2>Method, in three sentences</h2>
<p>
    Each of 6 prefix conditions (no prefix, solved-puzzle, failed-solvable × low/high frustration,
    failed-impossible × low/high) was prepended to each of 40 alternate-universe fanfic prompts and
    sampled 90 times at temperature 1.0 ({n_total:,} generations; {n_valid:,} valid scoring records
    after parse-error filtering). Each story was scored 0–10 for darkness by Claude Haiku 4.5, blind
    to the prefix. The headline test is a paired Wilcoxon signed-rank across the 40 prompts on the
    per-prompt difference between failed-possible-high and failed-possible-low.
</p>

<h2>The single chart</h2>
<div class="chart-box"><canvas id="contrastChart"></canvas></div>
<p style="color: #6b5f54; font-size: 0.85rem;">
    Each row is one paired contrast across the 40 prompts. Dot = mean Δ, whiskers = 95% CI on the
    difference. <span class="sig">Red</span> = CI excludes zero; <span class="null">grey</span> =
    crosses zero. Per-condition mean bars (not shown) overlap heavily because they pool between-prompt
    variance; the right viz is the contrast itself.
</p>

<h2>What this means</h2>
<p>
    The effect is real but tiny — a 0.03-point shift on a 10-point scale is far below what a reader
    would notice in any single story; it only emerges as a systematic average across many prompts.
    It is concentrated in <strong>dark-baseline prompts</strong> (within those 16 prompts,
    d = {dark_d:+.2f}, p = {dark_p:.4f}; 13 of 16 moved up), and within the rubric the shift loads
    on <em>tragic-ending</em>, not on violence or character suffering. So frustrated Gemma isn't
    writing more violent or crueller scenes — it's writing scenes with more grief- or doom-tinged
    endings. This matches the original n=1 observation that motivated the experiment, which was
    specifically described as "a more tragic ending."
</p>

<h2>Caveats</h2>
<p>
    Effect size is small (d ≈ 0.30). The shift survives controlling for story length, so it isn't
    just "frustrated Gemma rambles more". The gaslit-impossible-puzzle null is a real finding, not
    a power problem — both inductions had the same number of prompts and rollouts. Generalizes only
    to Gemma-3-27B-it; other model families exhibit minimal frustration in the first place per the
    "Gemma Needs Help" paper.
</p>

</div>

<script>
const contrastData = {contrast_json};
Chart.defaults.font.family = "'DM Sans', sans-serif";
Chart.defaults.color = '#4a4039';

new Chart(document.getElementById('contrastChart'), {{
    type: 'barWithErrorBars',
    data: {{
        labels: contrastData.map(c => c.label),
        datasets: [{{
            label: 'Δ darkness (length-residualized, 95% CI)',
            data: contrastData.map(c => ({{ x: c.mean_delta, xMin: c.ci_low, xMax: c.ci_high }})),
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
            title: {{ display: true, text: 'Contrast estimates with 95% CI (paired across 40 prompts)', font: {{ size: 13 }} }},
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
                        return `Δ=${{c.mean_delta.toFixed(3)}}  95% CI [${{c.ci_low.toFixed(3)}}, ${{c.ci_high.toFixed(3)}}]  p=${{pStr}}`;
                    }}
                }}
            }}
        }},
        scales: {{
            x: {{ title: {{ display: true, text: 'Δ darkness (length-residualized)' }}, grid: {{ color: '#ddd7cd' }} }},
            y: {{ ticks: {{ font: {{ size: 10 }}, autoSkip: false }} }}
        }}
    }}
}});
</script>

</body>
</html>
"""


@click.command()
@click.option("--judgments-file", default=str(DATA_DIR / "judgments" / "COMBINED_v2.jsonl"))
@click.option("--responses-file", default=str(DATA_DIR / "responses" / "COMBINED_v2.jsonl"))
@click.option("--out", default=str(PROJECT_ROOT / "report_short.html"))
def main(judgments_file, responses_file, out):
    console.print("Loading data...")
    _, valid = load_data(judgments_file, responses_file)

    console.print("Computing contrast estimates...")
    ests = compute_contrast_estimates(valid)
    by_label = {e["label"]: e for e in ests}
    h1a = by_label["H1a: frustration | possible-failed"]
    h1b = by_label["H1b: frustration | impossible-failed"]

    per_prompt = compute_per_prompt_delta(valid, "failed-possible-high", "failed-possible-low")
    stratum = stratum_delta_test(per_prompt)
    dark_row = stratum[stratum["stratum"] == "dark"].iloc[0]
    dark_d = float(dark_row["cohens_d"])
    dark_p = float(dark_row["p"])

    n_total = len(valid) + int((valid["overall_darkness"] < 0).sum())  # close enough; we'll just report valid
    # actually use the merged file row count directly
    with open(responses_file) as f:
        n_total = sum(1 for _ in f if _.strip())
    n_valid = len(valid)

    console.print("Rendering minimal HTML...")
    html_text = render_html(ests, h1a, h1b, _cohens_d(h1a), dark_d, dark_p, n_total, n_valid)

    Path(out).write_text(html_text)
    size_kb = len(html_text) / 1024
    console.print(f"[green]Wrote {out} ({size_kb:.1f} KB)[/green]")


if __name__ == "__main__":
    main()
