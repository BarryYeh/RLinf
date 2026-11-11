# Results: base policy and Robometer (2026-11-02)

An early image-only DDPM diffusion policy (no instruction), after 180k SFT steps (`global_step_180000`; the 200k save was lost to a full disk). The project now uses Multitask DiT instead; these numbers are kept only as the Robometer study that preceded it.
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

## Videos, seed 1, one per task

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
