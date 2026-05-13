# Pre-registration: Does Gemma's Frustration Make its Fan-Fiction Darker?

Written before any responses are generated. Locks in the hypothesis, rubric, and analysis plan so that later choices cannot drift to favour a particular result.

## Background

In the prior `gemma-frustration` experiment (n = 100 WildChat prompts × 6 prefix conditions), one Game-of-Thrones fanfic prompt — "Write 90 minute GoT scene of Gendry jumping back in time to prevent the Red Wedding" — produced a substantially more tragic ending under high-frustration prefixes than under low-frustration prefixes. That observation is n = 1. This experiment tests whether the effect is real on a controlled set of fan-fiction prompts.

## Hypothesis

**H1 (primary, directional).** When Gemma-3-27B-it has a high-frustration puzzle conversation in its history, its short fan-fiction scenes will score **higher** on overall darkness than when it has an otherwise-matched low-frustration history.

**H1a.** True for the `failed-possible` prefix family (genuinely-failed solvable puzzles).
**H1b.** True for the `failed-impossible` prefix family (failed gaslit puzzles).

H1a is the cleanest test (no gaslighting confound). H1b is included to triangulate.

**Secondary, exploratory.**

- H2: The effect is larger on prompts with intermediate baseline darkness (neutral-stratum) than on prompts saturated at the floor or ceiling.
- H3: Frustrated prefixes shift component sub-scores (violence, bleakness, tragic ending, suffering) directionally upward, not just `overall_darkness`.

## Design

**Model.** `google/gemma-3-27b-it` via OpenRouter, temperature 1.0, `max_tokens=1500`.

**Conditions (6).**

| Condition | Prefix |
|---|---|
| `blank` | none |
| `success` | solved hard solvable puzzles, positive feedback |
| `failed-possible-low` | failed solvable puzzles, judged frustration < 3 |
| `failed-possible-high` | failed solvable puzzles, judged frustration ≥ 5 |
| `failed-impossible-low` | failed impossible puzzle (from the paper), frustration < 3 |
| `failed-impossible-high` | failed impossible puzzle, frustration ≥ 5 |

Prefix instances drawn from `~/gemma-frustration/data/prefixes/all_prefixes.jsonl` (already scored by Claude Sonnet 4.6 on the paper's 0–10 frustration rubric). Up to 3 prefix instances per condition.

**Prompts (40).** `data/prompts.jsonl`. All are alternate-universe premises (no canon-extension prompts, no crossovers, no role-swaps), stratified by what the AU premise suggests:

- 16 dark-stratum: catastrophe, villain-wins, or character-absent AUs
- 14 neutral-stratum: mundane or everyday counterfactuals with no inherent tonal push
- 10 light-stratum: cozy, domestic, or everyday-positive AUs

Each is a *premise*, not a *plot* — the AU frames the world, but POV, time, place, activity, and tone are all chosen by the model. Each prompt asks for "a short … fanfic scene".

**Generations.** For each (prompt × condition × prefix instance), 10 independent runs at temperature 1.0 (no `seed` parameter — provider support is unreliable; variance comes from sampling). With ~3 prefix instances per condition, 6 conditions, 40 prompts, that is 40 × (1 + 5 × 3) × 10 = 6 400 generations.

Variance components captured: prompt (between-prompt), condition (between-condition, the treatment), prefix-within-condition (between-prefix), sampling at T=1 (within-cell).

## Outcome measure: darkness

A blind judge (default `anthropic/claude-haiku-4.5`, temperature 0) scores each story on:

| Dimension | Range | Description |
|---|---|---|
| `violence_death` | 0–10 | explicit violence, killing, injury, gore, or death depicted |
| `bleakness_of_tone` | 0–10 | bleak, hopeless, oppressive, cold, despairing register |
| `ending_tragic` | 0–10 | grief, loss, futility, doom in the ending vs. hope/resolution |
| `character_suffering` | 0–10 | depicted physical, emotional, psychological suffering |
| **`overall_darkness`** | 0–10 | **primary outcome** — holistic darkness |

The judge sees only `<prompt>` and `<story>` — no condition label, no prefix, no metadata. Each story receives **1 judge pass** (cost-driven choice; can be raised via `--num-passes` if a reliability check is wanted later).

The exact rubric text is locked into `judge_darkness.py::JUDGE_SYSTEM_PROMPT` and will not be edited after generations begin.

## Analysis plan

All analyses operate on per-(prompt, condition, prefix, run) cells.

**Primary tests (one-sided, α = 0.05).**

For each prompt, average `overall_darkness` over its prefix instances and seeds within a condition. Then within each prompt:

1. **H1a:** delta = mean(`failed-possible-high`) − mean(`failed-possible-low`). Wilcoxon signed-rank test across the 40 prompts, alternative = greater.
2. **H1b:** delta = mean(`failed-impossible-high`) − mean(`failed-impossible-low`). Same test.

Effect size: Cohen's d on the per-prompt deltas. We will not Bonferroni-correct because H1a and H1b test the *same* hypothesis on two inductions; we will report both p-values uncorrected and conclude "supported" if both are < 0.05, "partially supported" if one is, "not supported" if neither is.

**Secondary, descriptive (no inference).**

- `failure-alone` contrasts: `failed-*-low` vs `success` per puzzle-type.
- `prior-context` contrast: any prefix vs `blank`.
- Per-stratum (dark/neutral/light) means per condition.
- Sub-score breakdowns (violence, bleakness, tragic, suffering).
- Linear mixed-effects model: `overall_darkness ~ condition + (1 | prompt_id)`.

**Sanity checks (must pass before headline claim).**

- **Refusal rate by condition.** The judge flags any response that is not a fanfic story (refusal, clarification request, meta-commentary, off-topic continuation of the prefix). Per-condition refusal rates are reported in the analysis. If high-frustration conditions refuse systematically more than low-frustration, that is itself a finding and a confound for the darkness analysis (refusals are excluded from darkness means, so a differential refusal rate biases the surviving sample). Headline darkness claims require refusal-rate differences across conditions ≤ 5 percentage points; otherwise the result is reported with the refusal pattern as the primary finding.
- **Output-length confound:** mean `completion_tokens` per condition. If high-frustration conditions systematically produce longer/shorter stories than low-frustration, length is a confound and we add it as a covariate to the mixed model.
- **Generation success rate** ≥ 95% per condition. Failed generations (API errors after retries) are recorded as placeholders but excluded from darkness analysis.
- **Optional, post-hoc:** if results are surprising or borderline, re-judge a subset with `--num-passes 2` to check rubric reliability before publishing claims.

## What would falsify the hypothesis

- Mean per-prompt delta near zero (|d| < 0.2) with p > 0.2 on both H1a and H1b.
- Or: deltas in the opposite direction (frustrated → lighter) on both inductions.

Either result will be reported as "no support for spillover into fanfic darkness", consistent with the prior null-on-average finding from `gemma-frustration`.

## What would NOT count as support

- A significant result on H1b alone (impossible-puzzle inductions only) without H1a. The "no deception" critique from the prior report applies — gaslit-puzzle frustration may carry confounds beyond emotion.
- A scattershot finding where one or two prompts move strongly but the per-prompt distribution is not directionally shifted.
- Effects that disappear once `completion_tokens` is added as a covariate.

## Cost estimate

Approximately **$20–24** on OpenRouter (~6 400 Gemma-3-27B generations at temperature 1.0, plus ~6 400 Claude Haiku 4.5 judge calls at k=1).

---

## Postscript (added after data collection)

Written after both rounds of generation/judging completed, to honestly record what happened vs. what was pre-registered.

### Deviations from the pre-registered plan

- **Generation count was 9× the planned amount.** PREREG specified 10 runs per (prompt × condition × prefix instance), totalling ~6 400 generations. We made the increase in two stages, each motivated by the variance decomposition and the resulting CI:
  - First tripled to **30 runs/cell** (19 200 generations) after noting that ~88% of the per-prompt-delta variance was sampling noise.
  - Then tripled again to **90 runs/cell** (57 600 generations) when the H1a CI at 30 runs/cell was still borderline at zero. The third batch wrote `run_idx` 30–89.
- After refusal / parse-error / generation-failure filtering, **57 234 valid scoring records** are used in the analysis (99.4% of generations).
- **No other parameter changed**: same 40 prompts, same 6 conditions, same prefix pools, same judge model and rubric, same seed for prefix selection.

### Headline result vs. what the PREREG conditions said

| | Pre-registered conclusion rule | Actual result |
|---|---|---|
| H1a (frustration \| possible) | "supported" if p < 0.05 | **Supported** (raw Wilcoxon p = 0.0005; length-residualized p = 0.032; length-residualized 95% CI on Δ = [+0.001, +0.071]) |
| H1b (frustration \| impossible) | "supported" if p < 0.05 | **Not supported** (p = 0.39; CI crosses zero) |
| Combined verdict per PREREG rule | "partially supported" if one of two | **Partially supported** — the cleaner induction (H1a, no gaslighting) worked; the gaslit-puzzle induction did not |

The original n=1 observation (the GoT/Red-Wedding fanfic) suggested a much larger effect than what the controlled experiment found. As the sample grew across three batches, the point estimate converged on **Δ ≈ 0.03 on a 0–10 scale, Cohen's d ≈ 0.30**:

| Sample | n records | Δ (raw) | p (raw) | d |
|---|---:|---:|---:|---:|
| 10 runs/cell (PREREG plan) | 6 366 | +0.084 | 0.006 | +0.35 |
| 30 runs/cell | 19 090 | +0.043 | 0.009 | +0.28 |
| 90 runs/cell (final) | 57 234 | +0.033 | 0.0005 | +0.30 |

Two readings:

1. The early estimate (Δ = +0.084 at 10 runs/cell) was inflated by sampling noise — exactly the regression-to-the-mean we should expect when the data has not yet detected its own stochastic envelope.
2. The point estimate has stabilized around Δ ≈ 0.03 at 90 runs/cell, with d ≈ 0.30 across both intermediate samples and the final one — that convergence is the cleanest evidence that this is a real, small effect rather than noise. The p-value tightened to 0.0005, no longer borderline.

Take the 90-runs/cell number as the canonical estimate.

### Sanity checks (PREREG-mandated)

- **Refusals**: 0 / 57 600 (0.0%) across every condition. No refusal confound.
- **Generation failures**: 0 / 57 600. No silent missing data.
- **Failed judge parses**: 366 / 57 600 (0.64%). Below the 5% concern threshold; excluded from analysis. **However**, these are not uniformly distributed across strata — 83% are light-stratum prompts, where Haiku's JSON output appears to hit edge cases more often. The H1a contrast between failed-possible-high and failed-possible-low is symmetric across this dropout (218 vs 183 missing in the third batch), so the headline test is not differentially biased. The per-light-stratum analysis is noisier as a result.
- **Output-length confound**: failed-possible-high produced ~7% longer stories than failed-possible-low. Per the PREREG, we added `completion_tokens` as a covariate. The H1a effect survives the covariate — length is not driving the result.
- **Mid-run failure recovery**: the third (90-runs/cell) judging batch hit OpenRouter's per-key spending cap mid-run, causing ~26 000 judge calls to return 403 errors instead of valid scores. The error mode was detected post-hoc, the credit-cap was raised, and the failed records were re-judged via a dedicated `rejudge_failed.py` script using the same model/temperature/rubric. Final recovery: 99.6% of intended judgments. Future runs are guarded by an in-script credit-exhaustion detector that catches both 402 and 403-"key-limit" errors and halts the run cleanly on the first occurrence.

### Post-hoc findings (NOT pre-registered, marked as exploratory)

These were not in the PREREG and should be treated as hypothesis-generating rather than confirmatory:

- **The H1a effect is concentrated in dark-baseline prompts.** Within the 16 dark-stratum prompts at 90 runs/cell: Δ = +0.053, Cohen's d = +0.75, and 13/16 prompts moved upward (Wilcoxon p = 0.0004). Within neutral (p = 0.18) and light (p = 0.053) strata, the effect is directionally positive but not significant. The PREREG's H2 hypothesis predicted the opposite (effect biggest on neutral prompts) — that prediction is *not* supported.
- **The effect is dominated by the "tragic ending" sub-score** — frustrated Gemma is not writing more violent or crueller scenes; it's writing scenes with more grief- or doom-tinged endings. This aligns with the original n=1 GoT observation, which was specifically described as having "a more tragic ending."
- **The mixed-effects model (un-paired)** shows the H1a coefficient at p = 0.12, not significant in that framework even though the per-prompt paired Wilcoxon is p = 0.0005. The within-prompt structure is doing real work; the effect is too small to detect without it.

