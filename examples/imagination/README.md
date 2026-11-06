# Imagination SFT on LIBERO-Spatial

This folder holds the loop that improves a diffusion policy with rollouts imagined by the OpenSora world model (WM).
Each iteration collects real rollouts, imagines more from the same initial states, keeps the imagined rollouts that reward models score as successes, and fine-tunes the policy on them.

## Components

| Part | Code | Config |
|---|---|---|
| Policy: Multitask DiT (LeRobot `MultiTaskDiTPolicy`: CLIP frames + frozen CLIP instruction, DDPM; main camera, 2-frame history, 8-action chunks, no proprio state) | `rlinf/models/embodiment/multitask_dit/` | `examples/embodiment/config/model/multitask_dit.yaml` |
| Baseline policy: image-only DDPM diffusion policy (same inputs without the instruction) | `rlinf/models/embodiment/diffusion_policy/` | `examples/embodiment/config/model/diffusion_policy.yaml` |
| Policy SFT (`data.episodes_per_task` subsamples demos) | `rlinf/workers/sft/fsdp_vla_sft_worker.py` | `examples/sft/config/libero_spatial_sft_multitask_dit.yaml` |
| Policy eval in LIBERO | `evaluations/run_eval.sh`, `collect_rollouts.py --model-cfg` | `evaluations/libero/libero_spatial_multitask_dit_eval.yaml` |
| World model: action-conditioned OpenSora, 8 frames per 8-action chunk | `rlinf/envs/sim/world_model/` | `examples/embodiment/config/env/opensora_libero_spatial.yaml` |
| WM environment | `requirements/install_opensora_conda.sh` | — |

WM fine-tuning runs in the external [RLinf/opensora](https://github.com/RLinf/opensora) repository, not in RLinf.

## The loop

1. **Real rollouts.** Run the current policy in LIBERO on the training initial states, with K reruns per state.
   Keep failures as well as successes.
   Save, for each episode, the init-state id, the rerun index, the actions, the main-camera frames, and the success flag.
   These rollouts give the baseline success rate (SR), the WM fine-tuning data, and the ground truth for reward calibration.
2. **WM fine-tuning.** Fine-tune OpenSora on the step-1 rollouts so it matches LIBERO and the current policy's behavior, as in WMPO.
   Measure WM fidelity each iteration as the correlation (Pearson and MMRV) between imagined SR and real SR, per initial state.
3. **Imagination.** For each training initial state, start the WM from the real initial frame and roll the policy out N times in closed loop.
   The N rollouts form one group.
   Limit the horizon to the range where the WM stays reliable.
4. **Scoring and filtering.** Score every imagined rollout with RoboMeter and RoboReward.
   Keep a rollout only if both models call it a success.
   Then keep the top-k by mean score within its group.
   A group with no success contributes nothing.
5. **Policy fine-tuning.** The diffusion policy has no likelihood, so train it with filtered BC or reward-weighted SFT on the kept rollouts, as in Ctrl-World and VLAW.
   Mix in the original demos to prevent forgetting.
6. **Repeat** from step 1 with the new policy.

## Measuring reward errors

Measure reward false positives (FP) in two places:

- **Real frames.** Score the step-1 real videos with both reward models and compare against the simulator's success flags.
  This gives each model's FP rate and calibrates the success thresholds.
- **Imagined frames.** Each iteration, manually audit a random sample of kept imagined rollouts.
  Only this catches errors caused by WM artifacts.

Do not replay imagined actions in LIBERO to check rewards.
A policy acting on imagined frames does not reproduce the same trajectory in the real environment, so a failed replay usually means divergence, not a reward error.

## Fixed design decisions

Ask the project owner before changing any of these.

- **State split.** LIBERO-Spatial has 10 tasks with 50 initial states each.
  Train on 40 states per task and evaluate on the other 10.
  Store the split in config and never imagine from evaluation states.
- **Evaluation.** Report SR on the 100 held-out states, with K reruns per state.
- **Filtering.** An absolute success floor, then within-group top-k.
  Never rank alone, because the best of an all-failure group is still a failure.
  The floor needs both reward models, except in iteration 1, where Robometer alone judges (see below).
- **Training frames.** The policy trains on imagined frames and has no real-frame replay arm.

## Constraints in the current code

- **Initial frames.** The WM environment reads initial frames from `initial_image_path` in the WM checkpoint's dataset format.
  Step 1 must write real initial frames in that format, keyed by init-state id.
- **Seeds.** `DiffusionPolicy.sample_actions` is called without a generator, and `OpenSoraBackend` draws noise from the global RNG and ignores `seeds`.
  Rollouts in a group differ, but a single rollout cannot be reproduced from a seed, so save its actions.
- **Episode length.** The policy's frame history requires every episode to run for the full `max_episode_steps`, so set `ignore_terminations: True` in both LIBERO and the WM.
  Score rollouts after they finish, instead of using the WM's built-in `ResnetRM` termination.
- **Frame convention.** The WM's training frames must match the orientation and crop of the LIBERO frames the policy was trained on.
  Check this before WM fine-tuning.

## Iteration 1: Robometer judge, stock world model (2026-11-05)

Iteration 1 skips WM and reward-model fine-tuning.
The base policy is the Multitask DiT trained on 10 demos per task for 30k steps (ep10 @ 30k), and the WM is the stock OpenSora LIBERO-Spatial checkpoint.

- **Judge.** `robometer_judge.py` scores each episode in one Robometer query over 31 frames (steps 0, 8, …, 240).
  An episode succeeds when 3 scored frames in a row have a success probability above 0.5, and `success_step` is the first frame of that window.
  On the 400 real training episodes of trials 0–9 (4 reruns), it agrees with the simulator on 86% (recall 97%, precision 86%); most false positives are on task 3 (judge 57.5% vs simulator 22.5%).
- **Tasks.** Tasks whose imagined SR (this judge on closed-loop WM rollouts) differs from the real simulator SR by at least 30 points on trials 0–9:

  | Task | Real SR (sim) | Imagined SR (judge) | Gap |
  |---|---|---|---|
  | 1, bowl next to the ramekin | 72.5% | 15.0% | −57.5 |
  | 3, bowl on the cookie box | 22.5% | 87.5% | +65.0 |
  | 9, bowl on the wooden cabinet | 65.0% | 100.0% | +35.0 |

- **Imagination.** `imagine_rollouts.py --tasks 1,3,9 --trials 0-39 --reruns 8`: 8 closed-loop rollouts from each training state of those tasks.
- **Filtering.** Of each state's judged successes, `export_imagined.py` keeps the 4 with the highest mean success probability, each cut at `success_step`, as a LeRobot dataset.
- **Policy update.** Two arms fine-tune ep10 @ 30k for 10k steps (lr 1e-4 cosine, 500 warmup steps, batch 256): the 10 demos per task alone, and those demos with the kept rollouts at 50/50 per batch (`data.sampling_weights`).
- **Evaluation.** Base and both arms on held-out trials 40–49: tasks 1, 3, 9 with 8 reruns, and all tasks with 2 reruns.

Outputs are under `~/datastorage/runs/imagination/iter1/`, and the calibration is in `~/datastorage/runs/imagination/robometer_fidelity/judge_calibration.json`.

## Results: base policy and Robometer (2026-11-02)

The base policy is the diffusion policy after 180k SFT steps (`global_step_180000`; the 200k save was lost to a full disk).
Two runs of 100 episodes each cover the held-out states: 10 tasks × trials 40–49, one rollout per state, seeds 0 and 1.
`collect_rollouts.py` records the episodes and `score_robometer.py` scores them with [Robometer-4B](https://huggingface.co/lerobot/Robometer-4B), using 8 evenly spaced frames.

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
- **Implication for the loop.** Do not use Robometer as one of the two required success votes without fine-tuning it on LIBERO failures that include wrong-bowl placements. Within a successful episode, its progress curve rises steadily (median correlation with time 0.97–0.99), so it can still mark roughly where an episode succeeds.

Outputs are under `~/datastorage/runs/imagination/`: `eval_dp180k_trials40-49` (seed 0) and `eval_dp180k_trials40-49_ignoreterm` (seed 1, with videos).
Each holds `episodes/*.npz`, `summary.json`, and `robometer_summary_<episode-end>.json`.

### Videos, seed 1, one per task

All videos are in `~/datastorage/runs/imagination/eval_dp180k_trials40-49_ignoreterm/videos/` and run for the full 240 steps.
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
