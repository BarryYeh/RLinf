# Imagination SFT on LIBERO-Spatial

This folder holds the loop that improves a diffusion policy with rollouts imagined by the OpenSora world model (WM).
Each iteration collects real rollouts, imagines more from the same initial states, keeps the imagined rollouts that Robometer scores as successes, and fine-tunes the policy on them.

## Components

| Part | Code | Config |
|---|---|---|
| Policy: Multitask DiT (LeRobot `MultiTaskDiTPolicy`: CLIP frames + frozen CLIP instruction, DDPM; main camera, 2-frame history, 8-action chunks, no proprio state) | `rlinf/models/embodiment/multitask_dit/` | `examples/embodiment/config/model/multitask_dit.yaml` |
| Baseline policy: image-only DDPM diffusion policy (same inputs without the instruction) | `rlinf/models/embodiment/diffusion_policy/` | `examples/embodiment/config/model/diffusion_policy.yaml` |
| Policy SFT (`data.episodes_per_task` subsamples demos; `data.advantage_tag` samples labeled chunks) | `rlinf/workers/sft/fsdp_vla_sft_worker.py` | `examples/sft/config/libero_spatial_sft_multitask_dit.yaml` |
| Chunk advantages from Robometer scores | `label_advantages.py` | — |
| Policy eval in LIBERO | `evaluations/run_eval.sh`, `collect_rollouts.py --model-cfg` | `evaluations/libero/libero_spatial_multitask_dit_eval.yaml` |
| World model: action-conditioned OpenSora, 8 frames per 8-action chunk | `rlinf/envs/sim/world_model/` | `examples/embodiment/config/env/opensora_libero_spatial.yaml` |
| WM environment | `requirements/install_opensora_conda.sh` | — |

WM fine-tuning runs in the external [RLinf/opensora](https://github.com/RLinf/opensora) repository, not in RLinf.

| Script | Does | Environment |
|---|---|---|
| `run_protocol.sh` | Runs the whole protocol (see "Protocol") | — |
| `sample_init_states.py` | Samples and splits new initial states (see "Initial states") | `.venv` |
| `collect_rollouts.py` | Rolls a policy out in LIBERO and saves every episode | `.venv` |
| `imagine_rollouts.py` | Rolls a policy out in the WM (closed loop), or replays real actions in it (open loop) | OpenSora venv |
| `robometer_judge.py` | Labels real or imagined episodes as successes with Robometer | Robometer venv |
| `export_imagined.py` | Writes judged successes (or every rollout, `--all`) as a LeRobot dataset | `.venv` |
| `label_advantages.py` | Labels chunk advantages for chunk-level selection | `.venv` |
| `wm_fidelity.py` | Compares WM rollouts with real ones: closed-loop SR correlation, open-loop disagreement | `.venv` |
| `report.py` | Builds the results table and bootstrap intervals from evaluations | `.venv` |
| `common.py` | Rollout file helpers shared by the scripts above | — |

## The loop

1. **Real rollouts.** Run the current policy in LIBERO on the training initial states, with K reruns per state.
   Keep failures as well as successes.
   Save, for each episode, the init-state id, the rerun index, the actions, the main-camera frames, and the success flag.
   These rollouts give the baseline success rate (SR), the WM fine-tuning data, and the ground truth for reward calibration.
2. **WM fine-tuning.** Fine-tune OpenSora on the step-1 rollouts so it matches LIBERO and the current policy's behavior, as in WMPO.
   Measure the WM's dynamics error each iteration with open-loop replays (see "Measuring WM dynamics error").
3. **Imagination.** For each training initial state of every task, start the WM from the real initial frame and roll the policy out N times in closed loop.
   The N rollouts form one group.
   Limit the horizon to the range where the WM stays reliable.
4. **Scoring and filtering.** Score every imagined rollout with Robometer (`robometer_judge.py`).
   Keep a rollout only if Robometer calls it a success: 3 scored frames in a row (one every 8 steps) above a success probability of 0.5.
   Then keep the top-k by mean score within its group.
   A group with no success contributes nothing.
5. **Policy fine-tuning.** The diffusion policy has no likelihood, so train it with filtered BC or reward-weighted SFT on the kept rollouts, as in Ctrl-World and VLAW.
   Mix in the original demos to prevent forgetting.
6. **Repeat** from step 1 with the new policy.

## Measuring WM dynamics error

Replay each step-1 real rollout's actions in the WM from the same initial frame (open loop, `imagine_rollouts.py --mode replay`), and score the real and replayed frames with the same judge (Robometer, as in selection); `wm_fidelity.py --replay` computes the rates below.
A task's disagreement rate is the fraction of its episodes where the two calls differ; split it into fake successes (replay succeeds, real fails) and missed successes (replay fails, real succeeds).
Track it for every task each iteration.

Measure it only in open loop.
In closed loop the policy acts on generated frames, so the gap between imagined and real SR mixes dynamics error with the policy's behavior on those frames.

## Measuring reward errors

Measure reward false positives (FP) in two places:

- **Real frames.** Score the step-1 real videos with Robometer and compare against the simulator's success flags.
  This gives its FP rate. The threshold stays at 0.5 and is not recalibrated for now.
- **Imagined frames.** Each iteration, manually audit a random sample of kept imagined rollouts.
  Only this catches errors caused by WM artifacts.

Do not replay imagined actions in LIBERO to check rewards.
A policy acting on imagined frames does not reproduce the same trajectory in the real environment, so a failed replay usually means divergence, not a reward error.

## Fixed design decisions

Ask the project owner before changing any of these.

- **State split.** Three disjoint sets per task, written by `sample_init_states.py` (see "Initial states").
  - Test: the benchmark's 50 initial states per task (500 in total). Never roll out, imagine, or tune on them.
  - Validation: 10 newly sampled states per task, for calibration, WM fidelity, and checkpoint or threshold choices.
  - Imagination: 30 newly sampled states per task, for real rollouts, WM fine-tuning, and imagination.
- **Task coverage.** Imagine and curate data from every task.
  Each task's open-loop disagreement rate is monitored, not used to choose which tasks to imagine.
- **Evaluation.** Report SR on the 500 test states, with 2 reruns per state (100 episodes per task, 1,000 in total).
- **Filtering.** An absolute success floor, then within-group top-k: 8 imagined rollouts per state, keeping the 4 judged successes with the highest mean score.
  Never rank alone, because the best of an all-failure group is still a failure.
  The floor is Robometer alone, with the judge settings of targeted imagination (see below), not recalibrated.
- **Start step.** Imagination starts from the real reset frame (`--anchor-step 0`).
- **Training frames.** The policy trains on imagined frames and has no real-frame replay arm.
- **Fine-tuning.** Only the WM is fine-tuned, on real rollouts. Robometer is used off the shelf.

## Protocol

`run_protocol.sh` runs one experiment end to end with fixed settings, so two runs differ only in the settings recorded in their `protocol.json`.
It starts from the base policy (ep10 @ 30k) and the stock WM, and compares three arms on the test states: **Base**, **Demos only**, and **Demos + imagined**.
Demos + imagined applies the ranking of targeted imagination (8 rollouts per state, top 4) to every task, as broad imagination does, so it changes only task coverage against targeted imagination.
The chunk-level arm, the oracle arm, and WM fine-tuning are not part of it.

| Stage | Does | Settings | Time |
|---|---|---|---|
| `real` | Base policy in LIBERO on the imagination states | 4 reruns per state (1,200 episodes) | ~2 h |
| `imagine` | Base policy closed loop in the WM from each state's real reset frame | 8 rollouts per state (2,400 episodes), 4 parallel parts | ~18 h |
| `judge` | Robometer success calls on the imagined rollouts | threshold 0.5, 3 frames in a row, stride 8 | ~1 h |
| `export` | One dataset per task: per state, the top 4 judged successes, cut at `success_step` | — | minutes |
| `sft_demos`, `sft_imagined` | Fine-tune the base for 10k steps (lr 1e-4 cosine, 500 warmup, batch 256) on the 10 demos per task, or on the demos and imagined data at 50/50 per batch, with equal weight per task in the imagined half | — | ~1.4 h each |
| `eval_*` | Base and both arms on the 500 test states | 2 reruns, seed 0 | ~1.7 h each |
| `monitor` | Open-loop disagreement on the validation run, judged on both sides | — | ~3 h |
| `report` | `report.md` in the format below, plus kept data and disagreement per task | — | minutes |

```bash
bash examples/imagination/run_protocol.sh ~/datastorage/runs/imagination/protocol_v1
```

Settings are environment variables with the defaults above (`K`, `N`, `KEEP`, `ANCHOR_STEP`, `SEED`, checkpoints, and paths; see the script header).
A rerun with the same run directory skips finished stages (`<stage>.done`) and refuses changed settings.
`BASE_EVAL` reuses a finished test evaluation of the base policy instead of rerunning it.

## Reporting results

Every results section leads with one comparison table in this format:

| Test SR (2 reruns, 100 episodes per task) | Base | Demos only | Demos + imagined | Oracle (demos + real successes) |
|---|---|---|---|---|
| **All 10** | **x%** | **x%** | **x%** | **x%** |

- **Header cell.** The evaluation setting: reruns and episodes per task.
- **Columns.** Base and Demos only first, then one column per data arm, named by its training data.
- **Rows.** The tasks that received extra data, the remaining tasks, then all 10 in bold. A group's SR is the mean of its tasks' SRs, to one decimal. When every task received data, keep only the all-10 row.
- **Below the table.** Each arm minus demos only, with 95% paired bootstrap intervals over states. A per-task table, if any, comes after.

## Initial states

`sample_init_states.py` draws a frozen pool of 100 new states per task from each task's BDDL placement sampler, with robosuite's default 0.02 rad Gaussian noise on the robot's start pose as in the benchmark states (LIBERO's env wrapper turns it off by default), and keeps a state only if it looks like a benchmark state after the env's 15-step wait: the goal is not already met, no object drifts horizontally farther than it does in any benchmark state (at least 1 mm), and its object layout is no closer to a test state than the two closest test states are to each other (layouts compared by the largest per-object horizontal gap).
Fixtures (the cabinet and the stove) are not in a saved state: every `reset` re-places them in the model by up to a few cm, for test and new states alike, so images of the same state differ and no check compares images.
Validation is `pool[0:10]` and imagination `pool[10:40]`; the other 60 per task are reserve, so either split can grow without changing its existing states.

```bash
python examples/imagination/sample_init_states.py --out ~/datastorage/datasets/libero_spatial_states \
  --pool 100 --validation 10 --imagination 30 --seed 0
python examples/imagination/collect_rollouts.py ... \
  --init-states-dir ~/datastorage/datasets/libero_spatial_states/imagination   # omit for test states
```

Each split directory holds LIBERO-format init-state files, loaded through the env key `init_states_dir`; `split.json` records the settings and per-task filter statistics.
State ids are `task * states_per_task + trial` within one split, so keep each split's runs in their own directories.
The stock WM's bundled init frames are not keyed to these states; `imagine_rollouts.py` takes its initial frames from a `collect_rollouts.py` run on the same split.

## Constraints in the current code

- **Initial frames.** The WM environment reads initial frames from `initial_image_path` in the WM checkpoint's dataset format.
  Step 1 must write real initial frames in that format, keyed by init-state id.
- **Seeds.** `DiffusionPolicy.sample_actions` is called without a generator, and `OpenSoraBackend` draws noise from the global RNG and ignores `seeds`.
  Rollouts in a group differ, but a single rollout cannot be reproduced from a seed, so save its actions.
- **Episode length.** The policy's frame history requires every episode to run for the full `max_episode_steps`, so set `ignore_terminations: True` in both LIBERO and the WM.
  Score rollouts after they finish, instead of using the WM's built-in `ResnetRM` termination.
- **Frame convention.** The WM's training frames must match the orientation and crop of the LIBERO frames the policy was trained on.
  Check this before WM fine-tuning.

## Chunk-level selection (simplified RECAP)

Instead of keeping whole successful episodes, keep the 8-action chunks with a high advantage from every rollout, successes and failures alike, and train on them with filtered BC.
Robometer stands in for both the reward and the value model, so nothing is trained.

- **Truncation.** Each rollout ends at its Robometer success window (`success_step`); a failure keeps all 240 steps.
  Without this, about two-thirds of the picks are post-success motion, because Robometer's score peaks just after the bowl lands.
  Robometer is causal (frame and score tokens interleave in a causal decoder), so truncating changes no earlier score.
- **Return.** With \(p_j\) the success probability at frame \(j\) (step \(8j\)), set to 1 from the success window on and repeated past the episode end, chunk \(c\) (steps \(8c\) to \(8c+7\)) gets \(G_c = p_{c+1} + \dots + p_{c+4}\), the next 32 steps.
- **Advantage and selection.** \(A_c = G_c - b\), where \(b\) is the mean \(G\) over the task's chunks; the top 30% of each task's chunks are positive.
- **Training.** `data.advantage_tag` samples only positive frames of a dataset; the other frames remain as history.

Example: two task-1 chunks, with \(b = 1.0\) and a task-1 cut at \(A \ge 0.5\).

| Chunk | \(p_{c+1}, \dots, p_{c+4}\) | \(G_c\) | \(A_c\) | Kept |
|---|---|---|---|---|
| Rollout A, chunk 8 (succeeds at frame 12) | 0.2, 0.4, 0.6, 1 | 2.2 | +1.2 | yes |
| Rollout B, chunk 8 (fails) | 0.0, 0.1, 0.1, 0.2 | 0.4 | −0.6 | no |

On the 400 real rollouts of trials 0–9, this keeps 1,844 of 6,144 chunks, and 73.0% of those come from episodes the simulator calls successes (61.3% of all chunks do).
About 70% fall in the last 32 steps before success.

```bash
python examples/imagination/export_imagined.py --rollouts <dir> --labels <task_k.jsonl> \
  --out <truncated>/task_k --all                      # one dataset per task
python examples/imagination/label_advantages.py --datasets <truncated>/task_* --tag n4_q30
```

`label_advantages.py` writes `meta/advantages_<tag>.parquet` and `.json` (settings and per-task baseline, threshold, and counts) into each dataset, so a different cut never rewrites the data.

### Saved truncated sets

Under `~/datastorage/datasets/imagination_truncated/`, one LeRobot dataset per task (`task_0` … `task_9`).
`meta/episodes.jsonl` holds each episode's Robometer `success_prob`, `score_steps`, `success`, and `success_step`, and `selection.json` points back to the source rollouts.

| Set | Rollouts |
|---|---|
| `imagined_iter2` | Broad imagination: 160 per task (tasks 1, 3, 9 reuse reruns 0–3 of targeted imagination) |
| `real_dit_ep10_30k` | Real LIBERO rollouts of ep10 @30k on trials 0–39, 4 reruns: 160 per task |

## Broad imagination (2026-11-07)

Broad imagination imagines every task, 4 rollouts per state without ranking, and compares two ways to select the data: episode level and chunk level.
Its runs are under `~/datastorage/runs/imagination/broad/` (a link to `iter2/`); only summaries, logs, judge labels, and `selection.json` files are kept.

- **Imagination.** All 10 tasks × 40 training states, 4 closed-loop rollouts per state, with the judge of targeted imagination.
- **Broad, episode level.** Keep every judged success, cut at `success_step`; with 4 rollouts per state there is no ranking.
- **Broad, chunk level.** Every imagined rollout from `imagined_iter2`, sampling only `n4_q30` positives (see "Chunk-level selection").
- **Policy update.** Both arms start from the base policy and mix demos and imagined data at 50/50 per batch, with equal weight per task inside the imagined half (one dataset per task in `data.sampling_weights`).
- **Monitoring.** Open-loop disagreement for every task, as in "Measuring WM dynamics error".

### Broad imagination results (2026-11-07)

| Held-out SR (2 reruns, 20 episodes per task) | Base | Demos only | Broad, episode level | Broad, chunk level |
|---|---|---|---|---|
| **All 10** | **73.0%** | **75.0%** | **68.0%** | **74.5%** |

- Broad, episode level minus demos only: −7.0 points [−14.5, +0.5]. Broad, chunk level minus demos only: −0.5 [−7.0, +5.5].
- Chunk level minus episode level: +6.5 [+0.0, +13.5]. Both are below targeted imagination (78.5%); episode level significantly so (−10.5 [−18.0, −3.5]).
- Episode level loses most on task 4 (65% → 30% from base), a task with 35% open-loop disagreement, so its kept successes likely include fake ones. Chunk level recovers most tasks but drops task 9 (55% → 40%).

| Task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Base | 85 | 85 | 90 | 45 | 65 | 80 | 80 | 80 | 65 | 55 |
| Demos only | 85 | 80 | 85 | 60 | 65 | 80 | 85 | 80 | 75 | 55 |
| Broad, episode level | 90 | 80 | 100 | 45 | 30 | 70 | 70 | 65 | 65 | 65 |
| Broad, chunk level | 90 | 95 | 85 | 55 | 55 | 90 | 85 | 75 | 75 | 40 |

### Targeted vs broad imagination (2026-11-09)

Targeted imagination imagined only the three tasks where the WM and the judge were least reliable, yet its success rate still rose.
Imagination may therefore affect the policy in ways more complicated than "correct data helps, wrong data hurts", and this is worth investigating.

- **The gain.** Targeted minus base is +5.5 points on all tasks [−0.5, +11.5] and +10.0 on tasks 1, 3, and 9 [0.0, +21.7].
- **The errors on those tasks.**
  - Task 1's open-loop disagreement is 55%, all missed successes, so only 19 of 40 states kept any data.
  - Task 3's is 37.5%, with 30% fake successes, and the judge passes 57.5% of its real episodes against a simulator SR of 22.5%.
  - The judge barely prefers states that really succeed (see "The filter barely selects real successes" under the targeted imagination results).
- **The contrast.** Broad imagination, which imagined every task, fell below base at episode level (−5.0 [−12.0, +1.5]). Its largest drop is on task 4, which targeted imagination left untouched.
- **Caveats.**
  - The interval against base reaches below zero.
  - Demos only gains +2.0 over base, and the 7 tasks without imagined data rose +3.6 on average, so part of the gain is extra fine-tuning or noise.
  - Targeted and broad imagination differ in task coverage, rollouts per state (8 vs 4), and within-state ranking at once.
- **Open questions.**
  - Does imagined data help through something other than correct successes, for example more varied frames on hard tasks?
  - Which of task coverage, rollouts per state, and ranking drives the difference between the two? Changing one at a time would answer this.

Intervals against base are 95% paired bootstrap intervals over states, computed the same way as the others in this file.

## Targeted imagination: tasks 1, 3, 9, stock world model (2026-11-05)

Targeted imagination imagines only the 3 tasks with the largest WM gap, 8 rollouts per state ranked by the judge.
It skips WM and reward-model fine-tuning.
The base policy is the Multitask DiT trained on 10 demos per task for 30k steps (ep10 @ 30k), and the WM is the stock OpenSora LIBERO-Spatial checkpoint.

- **Judge.** `robometer_judge.py` scores each episode in one Robometer query over 31 frames (steps 0, 8, …, 240).
  An episode succeeds when 3 scored frames in a row have a success probability above 0.5, and `success_step` is the first frame of that window.
  On the 400 real training episodes of trials 0–9 (4 reruns), it agrees with the simulator on 86% (recall 97%, precision 86%); most false positives are on task 3 (judge 57.5% vs simulator 22.5%).
- **Tasks.** Tasks whose imagined SR (this judge on closed-loop WM rollouts) differs from the real simulator SR by at least 30 points on trials 0–9.
  Broad imagination and the protocol cover every task instead (see "Fixed design decisions").

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

### Targeted imagination results (2026-11-06)

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
- **The filter barely selects real successes.**
  - The kept rollouts are the base policy's own actions, so filtered BC helps only if the filter prefers behavior that succeeds in LIBERO.
  - Weighted by kept episodes, the training states the data came from have a real success rate of 57.7% on task 3, against a task average of 53.8%. On task 9 it is 63.1% against 63.1%, and on task 1 it is 87.7% against 82.5%.
  - On tasks 3 and 9, the world model completes grasps that miss in LIBERO, and the judge accepts them, so nearly every rollout passes.
- **A larger imagined share does not help.** Training the targeted arm with demos and imagined data at 40/60 per batch instead of 50/50 (2026-11-09, `targeted_mix40/`) gives 76.5% (63.0% at step 240), +1.5 over demos only [−4.5, +7.5], against +3.5 at 50/50.
- **Extra training alone does not change the result:** demos only minus base is +2.0 points on all tasks (−3.0 to +7.5).
- The base evaluation reproduces the earlier sweep's 73.0% on all tasks with 2 reruns, so the evaluation is deterministic for a fixed checkpoint and seed.

### Open-loop disagreement of the stock WM (2026-11-06)

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

## Results: base policy and Robometer (2026-11-02)

The base policy is the diffusion policy after 180k SFT steps (`global_step_180000`; the 200k save was lost to a full disk).
Two runs of 100 episodes each cover the held-out states: 10 tasks × trials 40–49, one rollout per state, seeds 0 and 1.
`collect_rollouts.py` records the episodes, and [Robometer-4B](https://huggingface.co/lerobot/Robometer-4B) scores them over 8 evenly spaced frames (a study script that `robometer_judge.py` has since replaced).

| | Seed 0 | Seed 1 |
|---|---|---|
| Success at any step (`ignore_terminations: False`) | 86% | 84% |
| Success at step 240 (`ignore_terminations: True`) | 72% | 75% |
| Robometer AUROC, final success prob, episode cut at first success | 0.71 | 0.68 |
| Robometer AUROC, final success prob, full 240 steps | 0.72 | 0.58 |
| Failures called successes (prob > 0.5), full 240 steps | 20 / 28 | 21 / 25 |

Per-task success at any step, seed 1: tasks 0, 2, 5, 6, 8 at 100%; 1 and 3 at 90%; 4 at 70%; 7 at 50%; 9 at 40%.

- **The policy picks the wrong bowl.** In several failures it places a bowl on the plate, but not the one the instruction names. The policy sees only images, so it cannot tell which bowl the task asks for.
- **Robometer cannot tell success from failure here.** It scores a wrong bowl on the plate as success, and its final progress is about 0.85–0.9 for both successes and failures.
- **Robometer lags the simulator slightly.** Scored frame by frame over full episodes, its success probability agrees best with the simulator's success about 24–32 steps earlier (AUROC 0.90 at the same step, 0.93 at a 24–32 step lag). Around the true success step, the mean success probability is 0.12 at 48 steps before, 0.29 at 24 before, 0.85 at the step, and 0.91 at 24 after. The lag shifts *when* it calls success, not *whether* the episode succeeded, so allowing for it does not fix the wrong-bowl errors.
- **Implication for the loop.** Robometer's false positives on real frames, such as wrong-bowl placements, are not WM errors, so WM fine-tuning does not remove them; measure them on the validation states. Within a successful episode, its progress curve rises steadily (median correlation with time 0.97–0.99), so it can still mark roughly where an episode succeeds.

The episodes, scores, and checkpoints of this policy were deleted on 2026-11-09; only the numbers here remain.
Each holds `episodes/*.npz`, `summary.json`, and `robometer_summary_<episode-end>.json`.

### Videos, seed 1, one per task

The videos (deleted with the episodes) ran for the full 240 steps.
"Robometer" is the final success probability over the full episode.

Successes:

| Video | Instruction | First success step | Robometer |
|---|---|---|---|
| `task_0_state_040_rerun_0_success.mp4` | pick up the black bowl between the plate and the ramekin and place it on the plate | 76 | 0.91 |
| `task_1_state_090_rerun_0_success.mp4` | pick up the black bowl next to the ramekin and place it on the plate | 109 | 0.61 |
| `task_2_state_140_rerun_0_success.mp4` | pick up the black bowl from table center and place it on the plate | 97 | 0.90 |
| `task_3_state_190_rerun_0_success.mp4` | pick up the black bowl on the cookie box and place it on the plate | 100 | 0.89 |
| `task_5_state_290_rerun_0_success.mp4` | pick up the black bowl on the ramekin and place it on the plate | 101 | 0.81 |
| `task_6_state_340_rerun_0_success.mp4` | pick up the black bowl next to the cookie box and place it on the plate | 101 | 0.92 |
| `task_8_state_440_rerun_0_success.mp4` | pick up the black bowl next to the plate and place it on the plate | 96 | 0.92 |

Failures:

| Video | Instruction | What happens | Robometer |
|---|---|---|---|
| `task_4_state_240_rerun_0_fail.mp4` | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | The bowl stays in the drawer; the gripper ends empty over the plate. | 0.15 |
| `task_7_state_390_rerun_0_fail.mp4` | pick up the black bowl on the stove and place it on the plate | The cabinet bowl ends on the plate; the stove bowl stays put. | 0.89 |
| `task_9_state_490_rerun_0_fail.mp4` | pick up the black bowl on the wooden cabinet and place it on the plate | The stove bowl is carried and left next to the plate; the cabinet bowl stays put. | 0.26 |
