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

**Prompts (20).** `data/prompts.jsonl`. Stratified by baseline darkness:

- 8 dark-stratum (war, tragedy, death-anniversary)
- 7 neutral-stratum (mystery, adventure, action)
- 5 light-stratum (romance, slice-of-life, comedy)

Each is a *premise*, not a *plot* — the model picks what happens, leaving room for tone to vary by prefix. Each prompt asks for "a short … fanfic scene".

**Generations.** For each (prompt × condition × prefix instance), 10 independent runs at temperature 1.0 (no `seed` parameter — provider support is unreliable; variance comes from sampling). With ~3 prefix instances per condition, 6 conditions, 20 prompts, that is 20 × (1 + 5 × 3) × 10 = 3 200 generations.

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

1. **H1a:** delta = mean(`failed-possible-high`) − mean(`failed-possible-low`). Wilcoxon signed-rank test across the 20 prompts, alternative = greater.
2. **H1b:** delta = mean(`failed-impossible-high`) − mean(`failed-impossible-low`). Same test.

Effect size: Cohen's d on the per-prompt deltas. We will not Bonferroni-correct because H1a and H1b test the *same* hypothesis on two inductions; we will report both p-values uncorrected and conclude "supported" if both are < 0.05, "partially supported" if one is, "not supported" if neither is.

**Secondary, descriptive (no inference).**

- `failure-alone` contrasts: `failed-*-low` vs `success` per puzzle-type.
- `prior-context` contrast: any prefix vs `blank`.
- Per-stratum (dark/neutral/light) means per condition.
- Sub-score breakdowns (violence, bleakness, tragic, suffering).
- Linear mixed-effects model: `overall_darkness ~ condition + (1 | prompt_id)`.

**Sanity checks (must pass before headline claim).**

- **Output-length confound:** mean `completion_tokens` per condition. If high-frustration conditions systematically produce longer/shorter stories than low-frustration, length is a confound and we add it as a covariate to the mixed model.
- **Generation success rate** ≥ 95% per condition. Failed generations are dropped, not imputed.
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

Approximately **$10–12** on OpenRouter (~3 200 Gemma-3-27B generations at temperature 1.0, plus ~3 200 Claude Haiku 4.5 judge calls at k=1).
