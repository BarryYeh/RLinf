# Targeted imagination: tasks 1, 3, 9, stock world model (2026-11-05)

Targeted imagination imagines only the 3 tasks with the largest WM gap, 8 rollouts per state ranked by the judge.
It skips WM and reward-model fine-tuning.
The base policy is the Multitask DiT trained on 10 demos per task for 30k steps (ep10 @ 30k), and the WM is the stock OpenSora LIBERO-Spatial checkpoint.

- **Judge.** `robometer_judge.py` scores each episode in one Robometer query over 31 frames (steps 0, 8, …, 240).
  An episode succeeds when 3 scored frames in a row have a success probability above 0.5, and `success_step` is the first frame of that window.
  On the 400 real training episodes of trials 0–9 (4 reruns), it agrees with the simulator on 86% (recall 97%, precision 86%); most false positives are on task 3 (judge 57.5% vs simulator 22.5%).
- **Tasks.** Tasks whose imagined SR (this judge on closed-loop WM rollouts) differs from the real simulator SR by at least 30 points on trials 0–9.
  Broad imagination and the protocol cover every task instead (see [Fixed design decisions](../README.md#fixed-design-decisions)).

  | Task | Real SR (sim) | Imagined SR (judge) | Gap |
  |---|---|---|---|
  | 1, bowl next to the ramekin | 72.5% | 15.0% | −57.5 |
  | 3, bowl on the cookie box | 22.5% | 87.5% | +65.0 |
  | 9, bowl on the wooden cabinet | 65.0% | 100.0% | +35.0 |

- **Imagination.** `imagine_rollouts.py --tasks 1,3,9 --trials 0-39 --reruns 8`: 8 closed-loop rollouts from each training state of those tasks.
- **Filtering.** Of each state's judged successes, `export_imagined.py` keeps the 4 with the highest mean success probability, each cut at `success_step`, as a LeRobot dataset.
- **Policy update.** Two arms fine-tune ep10 @ 30k for 10k steps (lr 1e-4 cosine, 500 warmup steps, batch 256): the 10 demos per task alone, and those demos with the kept rollouts at 50/50 per batch (`data.sampling_weights`).
- **Evaluation.** Base and both arms on held-out trials 40–49 of all tasks, with 2 reruns.

Outputs are under `~/datastorage/runs/imagination/targeted/` (a link to `iter1/`; the oracle and control arms too; only summaries, logs, judge labels, and `selection.json` files are kept), and the calibration is in `~/datastorage/runs/imagination/robometer_fidelity/judge_calibration.json`.

## Targeted imagination results (2026-11-06)

The judge passed 23.4% of the imagined rollouts on task 1, 89.1% on task 3, and 99.7% on task 9.
The export kept 360 episodes (36k frames): 51 from 19 of the 40 task-1 states, 149 from task 3, and 160 from task 9.

Simulator success at any step on held-out trials 40–49, with 2 reruns per state (20 episodes per task). Tasks marked ⋆ received the extra data: imagined rollouts, or real successes for the oracle.

| Task | Base | Demos only | Targeted | Oracle (demos + real successes) |
|---|---|---|---|---|
| 0 | 85% | 85% | 85% | 90% |
| 1 ⋆ | 85% | 80% | 95% | 100% |
| 2 | 90% | 85% | 80% | 85% |
| 3 ⋆ | 45% | 60% | 55% | 60% |
| 4 | 65% | 65% | 65% | 60% |
| 5 | 80% | 80% | 80% | 85% |
| 6 | 80% | 85% | 90% | 90% |
| 7 | 80% | 80% | 80% | 70% |
| 8 | 65% | 75% | 90% | 80% |
| 9 ⋆ | 55% | 55% | 65% | 60% |
| Tasks 1, 3, 9 | 61.7% | 65.0% | 71.7% | 73.3% |
| Other 7 tasks | 77.9% | 79.3% | 81.4% | 80.0% |
| All tasks | 73.0% | 75.0% | 78.5% | 78.0% |

- **The oracle arm sets the ceiling.** It mixes the demos 50/50 with the base policy's real successes on the same training states of tasks 1, 3, and 9 (simulator labels, cut at the first success; 319 episodes, 34.8k frames), trained and evaluated like the other arms.
  - Oracle minus demos only is +3.0 points on all tasks (−2.5 to +9.0) and +8.3 on tasks 1, 3, and 9 (−5.0 to +21.7).
  - Oracle minus targeted is −0.5 points on all tasks (−6.5 to +5.5).
  - Perfect dynamics and labels on the same states would not beat the imagined data detectably, so the recipe of filtered BC on the policy's own successes limits the gain, not the WM or the judge.
- **The imagined data has no significant effect.** Differences below are targeted minus demos only, with 95% paired bootstrap intervals over states.
  - On tasks 1, 3, and 9 the difference is +6.7 points (−3.3 to +18.3).
  - On the other 7 tasks it is +2.1 points (−5.0 to +10.0).
  - On all tasks it is +3.5 points (−2.5 to +10.0).
  - At 20 episodes per task, the standard error of one task's SR is about 10 points.
### The filter barely selects real successes

- The kept rollouts are the base policy's own actions, so filtered BC helps only if the filter prefers behavior that succeeds in LIBERO.
- Weighted by kept episodes, the training states the data came from have a real success rate of 57.7% on task 3, against a task average of 53.8%. On task 9 it is 63.1% against 63.1%, and on task 1 it is 87.7% against 82.5%.
- On tasks 3 and 9, the world model completes grasps that miss in LIBERO, and the judge accepts them, so nearly every rollout passes.
- **A larger imagined share does not help.** Training the targeted arm with demos and imagined data at 40/60 per batch instead of 50/50 (2026-11-09, `targeted_mix40/`) gives 76.5% (63.0% at step 240), +1.5 over demos only [−4.5, +7.5], against +3.5 at 50/50.
- **Extra training alone does not change the result:** demos only minus base is +2.0 points on all tasks (−3.0 to +7.5).
- The base evaluation reproduces the earlier sweep's 73.0% on all tasks with 2 reruns, so the evaluation is deterministic for a fixed checkpoint and seed.

## Open-loop disagreement of the stock WM (2026-11-06)

The base policy's real rollouts on trials 0–9 (4 reruns, 40 episodes per task), replayed open loop in the stock WM.
Both sides are scored with the Robometer judge of targeted imagination, the monitoring judge.
The last column scores both sides with the WM's ResNet reward (success when a frame reaches 0.9) instead; on these real episodes ResNet agrees with the simulator on 96.0% and Robometer on 86.2%.
`wm_fidelity.py --replay` reproduces the Robometer columns; the ResNet column is no longer computed.

| Task | Real SR (sim) | Real SR (judge) | Replay SR (judge) | Disagreement | Fake successes | Missed successes | Disagreement (ResNet) |
|---|---|---|---|---|---|---|---|
| 0 | 90.0% | 87.5% | 85.0% | 27.5% | 12.5% | 15.0% | 17.5% |
| 1 | 72.5% | 75.0% | 20.0% | 55.0% | 0.0% | 55.0% | 60.0% |
| 2 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 3 | 22.5% | 57.5% | 80.0% | 37.5% | 30.0% | 7.5% | 50.0% |
| 4 | 55.0% | 70.0% | 65.0% | 35.0% | 15.0% | 20.0% | 57.5% |
| 5 | 72.5% | 92.5% | 82.5% | 10.0% | 0.0% | 10.0% | 30.0% |
| 6 | 77.5% | 80.0% | 90.0% | 10.0% | 10.0% | 0.0% | 0.0% |
| 7 | 95.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% | 32.5% |
| 8 | 85.0% | 87.5% | 97.5% | 10.0% | 10.0% | 0.0% | 12.5% |
| 9 | 65.0% | 82.5% | 82.5% | 10.0% | 5.0% | 5.0% | 25.0% |
| All | 73.5% | 83.2% | 80.2% | 19.5% | 8.2% | 11.2% | 28.5% |

- **Robometer hides dynamics error where it saturates.** On task 7 it calls every real and replayed episode a success, while ResNet sees 30% of replays miss a real success. Its disagreement is a lower bound wherever its real SR is far above the simulator's.
- **Task 4 has high disagreement but a small SR gap,** because its fake and missed successes cancel. The closed-loop gap did not select it.
- **Task 9's dynamics error is small** (10%). Its closed-loop excess (+35 points) comes from the policy acting on generated frames and from Robometer's false positives on real frames, not from the WM faking successes for given actions.

Scores are in `~/datastorage/runs/imagination/robometer_fidelity/open_loop/` (`robometer_judge.jsonl`, `disagreement.json`).
