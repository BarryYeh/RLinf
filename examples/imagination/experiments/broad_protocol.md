# Broad imagination under the protocol (2026-11-10)

This run repeats broad imagination at episode level with `run_protocol.sh`, on the new state split and the 500 test states.
Its directory is `~/datastorage/runs/imagination/protocol_broad_episode/`, with the settings in `protocol.json` (commit `e4fd25f2`) and the full report in `report.md`.

- **Settings.** The protocol defaults, except `N=4`: 4 real reruns and 4 imagined rollouts per imagination state, keeping every judged success (`KEEP=4`, so no ranking), anchor step 0.
- **Data.** Real rollouts succeed in 76.0% of 1,200 episodes, and the judge passes 81.3% of the 1,200 imagined ones. Export keeps 976 episodes.

| Test SR (2 reruns, 100 episodes per task) | Base | Demos only | Demos + imagined |
|---|---|---|---|
| **All 10** | **72.4%** | **73.5%** | **72.2%** |

- Demos + imagined minus demos only: −1.3 points [−4.7, +2.1] at any step, and −3.1 [−6.9, +0.8] at step 240 (60.3%, 61.3%, 58.2%).
- The [broad episode-level arm of 2026-11-07](broad.md) lost 7.0 points against demos only. The two runs differ in states, test set, and number of test episodes, so the smaller loss here is not a paired comparison.
- Imagined data helps tasks 4 and 9 most (+10 and +9 over demos only) and hurts tasks 1 and 7 most (−9 each). On task 7 the loss is larger at step 240 (71% → 48%).

| Task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Base | 81 | 85 | 87 | 52 | 51 | 82 | 81 | 84 | 61 | 60 |
| Demos only | 81 | 87 | 91 | 55 | 49 | 84 | 80 | 83 | 69 | 56 |
| Demos + imagined | 79 | 78 | 85 | 57 | 59 | 84 | 79 | 74 | 62 | 65 |
| Real SR, imagination states | 78 | 88 | 79 | 62 | 62 | 83 | 88 | 80 | 63 | 78 |
| Judged imagined SR | 74 | 23 | 100 | 89 | 55 | 93 | 100 | 88 | 93 | 99 |
| Kept episodes | 89 | 27 | 120 | 107 | 66 | 112 | 120 | 105 | 111 | 119 |

The open-loop disagreement on the validation run matches the [2026-11-06 measurement](targeted.md#open-loop-disagreement-of-the-stock-wm-2026-11-06) (all tasks 19.5%, 8.2%, 11.2%):

| Task | Real SR (sim) | Real SR (judge) | Replay SR (judge) | Disagreement | Fake successes | Missed successes |
|---|---|---|---|---|---|---|
| 0 | 77.5% | 90.0% | 87.5% | 12.5% | 5.0% | 7.5% |
| 1 | 85.0% | 90.0% | 40.0% | 60.0% | 5.0% | 55.0% |
| 2 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| 3 | 77.5% | 90.0% | 97.5% | 7.5% | 7.5% | 0.0% |
| 4 | 42.5% | 52.5% | 57.5% | 55.0% | 30.0% | 25.0% |
| 5 | 77.5% | 92.5% | 65.0% | 27.5% | 0.0% | 27.5% |
| 6 | 95.0% | 97.5% | 100.0% | 2.5% | 2.5% | 0.0% |
| 7 | 57.5% | 97.5% | 100.0% | 2.5% | 2.5% | 0.0% |
| 8 | 65.0% | 90.0% | 92.5% | 2.5% | 2.5% | 0.0% |
| 9 | 62.5% | 82.5% | 95.0% | 12.5% | 12.5% | 0.0% |
| All | 74.0% | 88.2% | 83.5% | 18.2% | 6.8% | 11.5% |

- **Task 1.** The WM misses 55% of real successes, so the judge passes 23% of imagined rollouts and only 13 of 30 states keep data. Task 1 is also one of the two tasks that lose most.
- **Task 4.** It has the most fake successes (30%), yet gains the most. This repeats the [targeted imagination](targeted.md) puzzle: wrong data on a task does not predict a loss on it.
- **Task 7.** The error is in the judge, not the WM: it passes 97.5% of real validation episodes that succeed 57.5% of the time in the simulator. Its 105 kept episodes likely include failures, which may explain the loss at step 240. This is unchecked; replaying the kept task 7 episodes' actions in LIBERO would test it.
