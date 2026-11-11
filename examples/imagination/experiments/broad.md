# Broad imagination (2026-11-07)

Broad imagination imagines every task, 4 rollouts per state without ranking, and compares two ways to select the data: episode level and chunk level.
Its runs are under `~/datastorage/runs/imagination/broad/` (a link to `iter2/`); only summaries, logs, judge labels, and `selection.json` files are kept.

- **Imagination.** All 10 tasks × 40 training states, 4 closed-loop rollouts per state, with the judge of [targeted imagination](targeted.md).
- **Broad, episode level.** Keep every judged success, cut at `success_step`; with 4 rollouts per state there is no ranking.
- **Broad, chunk level.** Every imagined rollout from `imagined_iter2`, sampling only `n4_q30` positives (see [Chunk-level selection](../README.md#chunk-level-selection-simplified-recap)).
- **Policy update.** Both arms start from the base policy and mix demos and imagined data at 50/50 per batch, with equal weight per task inside the imagined half (one dataset per task in `data.sampling_weights`).
- **Monitoring.** Open-loop disagreement for every task, as in [Measuring WM dynamics error](../README.md#measuring-wm-dynamics-error).

## Broad imagination results (2026-11-07)

| Held-out SR (2 reruns, 20 episodes per task) | Base | Demos only | Broad, episode level | Broad, chunk level |
|---|---|---|---|---|
| **All 10** | **73.0%** | **75.0%** | **68.0%** | **74.5%** |

- Broad, episode level minus demos only: −7.0 points [−14.5, +0.5]. Broad, chunk level minus demos only: −0.5 [−7.0, +5.5].
- Chunk level minus episode level: +6.5 [+0.0, +13.5]. Both are below [targeted imagination](targeted.md) (78.5%); episode level significantly so (−10.5 [−18.0, −3.5]).
- Episode level loses most on task 4 (65% → 30% from base), a task with 35% open-loop disagreement, so its kept successes likely include fake ones. Chunk level recovers most tasks but drops task 9 (55% → 40%).

| Task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Base | 85 | 85 | 90 | 45 | 65 | 80 | 80 | 80 | 65 | 55 |
| Demos only | 85 | 80 | 85 | 60 | 65 | 80 | 85 | 80 | 75 | 55 |
| Broad, episode level | 90 | 80 | 100 | 45 | 30 | 70 | 70 | 65 | 65 | 65 |
| Broad, chunk level | 90 | 95 | 85 | 55 | 55 | 90 | 85 | 75 | 75 | 40 |

## Targeted vs broad imagination (2026-11-09)

Targeted imagination imagined only the three tasks where the WM and the judge were least reliable, yet its success rate still rose.
Imagination may therefore affect the policy in ways more complicated than "correct data helps, wrong data hurts", and this is worth investigating.

- **The gain.** Targeted minus base is +5.5 points on all tasks [−0.5, +11.5] and +10.0 on tasks 1, 3, and 9 [0.0, +21.7].
- **The errors on those tasks.**
  - Task 1's open-loop disagreement is 55%, all missed successes, so only 19 of 40 states kept any data.
  - Task 3's is 37.5%, with 30% fake successes, and the judge passes 57.5% of its real episodes against a simulator SR of 22.5%.
  - The judge barely prefers states that really succeed (see [The filter barely selects real successes](targeted.md#the-filter-barely-selects-real-successes) under the targeted imagination results).
- **The contrast.** Broad imagination, which imagined every task, fell below base at episode level (−5.0 [−12.0, +1.5]). Its largest drop is on task 4, which targeted imagination left untouched.
- **Caveats.**
  - The interval against base reaches below zero.
  - Demos only gains +2.0 over base, and the 7 tasks without imagined data rose +3.6 on average, so part of the gain is extra fine-tuning or noise.
  - Targeted and broad imagination differ in task coverage, rollouts per state (8 vs 4), and within-state ranking at once.
- **Open questions.**
  - Does imagined data help through something other than correct successes, for example more varied frames on hard tasks?
  - Which of task coverage, rollouts per state, and ranking drives the difference between the two? Changing one at a time would answer this.

Intervals against base are 95% paired bootstrap intervals over states, computed the same way as the others in these experiment notes.
