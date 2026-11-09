# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Roll out a frame-history policy in LIBERO and save every episode with its init state.

``--model-cfg`` selects the policy config (``diffusion_policy`` or
``multitask_dit``); ``--ckpt`` is its SFT ``full_weights.pt``.

Each episode is written to ``<out>/episodes/state_<id>_rerun_<k>.npz`` with the
main-camera frames (the reset frame plus one per step), the executed actions, the
simulator's per-step goal check (``step_success``), and the LIBERO task, trial, and
init-state ids. ``summary.json`` holds the success rates.

Every episode runs to ``max_episode_steps``. ``first_success_step`` marks where it
would end with ``ignore_terminations: False`` and ``success_at_end`` is its outcome
with ``True``; ``--ignore-terminations`` sets the env flag itself.
``--videos-per-task`` writes that many mp4s per task to ``<out>/videos``.
"""

import argparse
import os
from pathlib import Path

import imageio
import numpy as np
import torch
from common import load_policy, parse_ids, save_episode, write_summary
from omegaconf import OmegaConf

from rlinf.utils.logging import get_logger

logger = get_logger()

REPO_PATH = Path(__file__).resolve().parents[2]
ENV_CFG = REPO_PATH / "examples/embodiment/config/env/libero_spatial.yaml"
DEFAULT_MODEL_CFG = REPO_PATH / "examples/embodiment/config/model/diffusion_policy.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="Policy full_weights.pt.")
    parser.add_argument(
        "--model-cfg",
        default=str(DEFAULT_MODEL_CFG),
        help="Model config YAML the checkpoint was trained with.",
    )
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument(
        "--trials",
        default=None,
        help="Trial ids per task, as 'a-b' (inclusive) or a comma list; "
        "default: every state.",
    )
    parser.add_argument(
        "--init-states-dir",
        default=None,
        help="A split directory written by sample_init_states.py; "
        "default: the benchmark (test) states.",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="Task ids, as 'a-b' (inclusive) or a comma list; default: every task.",
    )
    parser.add_argument(
        "--reruns",
        type=int,
        default=2,
        help="Rollouts per state; test evaluation uses 2.",
    )
    parser.add_argument("--num-envs", type=int, default=50)
    parser.add_argument("--max-episode-steps", type=int, default=240)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ignore-terminations", action="store_true")
    parser.add_argument("--videos-per-task", type=int, default=0)
    parser.add_argument("--video-fps", type=int, default=20)
    return parser.parse_args()


def build_env(args: argparse.Namespace):
    from rlinf.envs.sim.libero.libero_env import LiberoEnv

    cfg = OmegaConf.load(ENV_CFG)
    cfg.ignore_terminations = args.ignore_terminations
    cfg.auto_reset = False
    cfg.is_eval = False
    cfg.use_fixed_reset_state_ids = False
    cfg.max_episode_steps = args.max_episode_steps
    cfg.max_steps_per_rollout_epoch = args.max_episode_steps
    cfg.seed = args.seed
    cfg.init_states_dir = args.init_states_dir
    cfg.video_cfg.save_video = False
    return LiberoEnv(
        cfg,
        num_envs=args.num_envs,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    num_actions = OmegaConf.load(args.model_cfg).num_action_chunks
    if args.max_episode_steps % num_actions != 0:
        raise ValueError(
            f"Expected --max-episode-steps divisible by num_action_chunks="
            f"{num_actions}, got {args.max_episode_steps}."
        )
    num_chunks = args.max_episode_steps // num_actions

    env = build_env(args)
    policy = load_policy(args.model_cfg, args.ckpt, num_chunks)

    bins = np.asarray(env.trial_id_bins)
    trials = parse_ids(args.trials) if args.trials else list(range(bins.max()))
    tasks = parse_ids(args.tasks) if args.tasks else list(range(len(bins)))
    starts = np.concatenate([[0], np.cumsum(bins)[:-1]])
    jobs = [
        (int(starts[task] + trial), task, trial, rerun)
        for rerun in range(args.reruns)
        for trial in trials
        for task in tasks
        if trial < bins[task]
    ]
    logger.info(
        f"{len(jobs)} episodes: tasks {tasks} x trials {trials} x "
        f"{args.reruns} reruns, {args.num_envs} envs per batch"
    )

    records = []
    for batch_start in range(0, len(jobs), args.num_envs):
        batch = jobs[batch_start : batch_start + args.num_envs]
        # The env needs a full batch; padding slots repeat the first job and are dropped.
        padded = batch + [batch[0]] * (args.num_envs - len(batch))
        env.is_start = False
        obs, _ = env.reset(reset_state_ids=np.array([job[0] for job in padded]))
        policy.reset()

        frames = [obs["main_images"].numpy().copy()]
        actions, success = [], []
        for _ in range(num_chunks):
            chunk, _ = policy.predict_action_batch(obs)
            for i in range(num_actions):
                step_action = chunk[:, i].numpy()
                obs, _, terminations, _, infos = env.step(step_action, auto_reset=False)
                # With ignore_terminations the env zeroes terminations and keeps
                # the goal check in success_at_end.
                if args.ignore_terminations:
                    terminations = infos["episode"]["success_at_end"]
                frames.append(obs["main_images"].numpy().copy())
                actions.append(step_action)
                success.append(np.asarray(terminations).copy())

        frames = np.stack(frames, axis=1)  # [B, T + 1, H, W, 3]
        actions = np.stack(actions, axis=1)  # [B, T, 7]
        success = np.stack(success, axis=1)  # [B, T]
        for slot, (state_id, task, trial, rerun) in enumerate(batch):
            hits = np.flatnonzero(success[slot])
            first = int(hits[0]) + 1 if len(hits) else -1
            record = {
                "state_id": state_id,
                "task_id": task,
                "trial_id": trial,
                "rerun": rerun,
                "task": env.task_descriptions[slot],
                "success": bool(len(hits)),
                "first_success_step": first,
                "success_at_end": bool(success[slot, -1]),
            }
            save_episode(
                out,
                record,
                frames=frames[slot],
                actions=actions[slot],
                step_success=success[slot],
            )
            records.append(record)
            num_task_videos = sum(
                1 for r in records[:-1] if r["task_id"] == task and r.get("video")
            )
            if num_task_videos < args.videos_per_task:
                video = (
                    out
                    / "videos"
                    / (
                        f"task_{task}_state_{state_id:03d}_rerun_{rerun}_"
                        f"{'success' if record['success'] else 'fail'}.mp4"
                    )
                )
                video.parent.mkdir(exist_ok=True)
                imageio.mimwrite(video, frames[slot], fps=args.video_fps)
                record["video"] = str(video.relative_to(out))
        done = len(records)
        rate = np.mean([r["success"] for r in records])
        logger.info(f"{done}/{len(jobs)} episodes, success rate so far {rate:.3f}")

    summary = write_summary(
        out,
        records,
        ckpt=os.path.abspath(args.ckpt),
        model_cfg=os.path.abspath(args.model_cfg),
        init_states_dir=args.init_states_dir,
        trials=trials,
        reruns=args.reruns,
    )
    logger.info(
        f"Success rate {summary['success_rate']:.3f} over {len(records)} episodes "
        f"(success at step {args.max_episode_steps}: "
        f"{summary['success_at_end_rate']:.3f})"
    )
    env.env.close()


if __name__ == "__main__":
    main()
