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

"""Roll out episodes in the OpenSora world model from frames of real LIBERO rollouts.

``--real`` is a finished ``collect_rollouts.py`` run. An episode starts from a real
episode's frame at ``--anchor-step`` (0 is the reset frame) and runs to
``--max-episode-steps`` of the real episode. That frame is the world model's initial
frame, written to ``<out>/init_frames/`` in its trajectory format; the condition
window repeats it with the env's reset condition actions, at any anchor.

``--mode policy`` runs the policy closed loop in the world model, ``--reruns`` times
per state, from the state's rerun 0. With ``--anchors`` (an ``anchor/select.py``
output) each state instead has a list of start points, each a real rerun and its own
anchor step, and imagined rerun ``k`` starts from point ``k`` modulo their number.
``--mode replay`` feeds each real episode's actions from the anchor on to the world
model open loop, starting from that episode's own frame; it measures the dynamics
without the policy reacting to generated frames.

Each episode is written to ``<out>/episodes/state_<id>_rerun_<k>.npz`` with the
generated frames (the initial frame plus one per step), the actions, the world
model's reward-model success probability (``rm_reward``) and call (``step_success``:
``rm_reward >= --success-threshold``), ``anchor_step``, ``source_rerun`` (the real
rerun it starts from), and the LIBERO task, trial and init-state ids.
``summary.json`` holds the success rates. Run it in the OpenSora environment.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from common import (
    episode_name,
    load_policy,
    load_records,
    parse_ids,
    save_episode,
    write_summary,
)
from omegaconf import OmegaConf

from rlinf.utils.logging import get_logger

logger = get_logger()

REPO_PATH = Path(__file__).resolve().parents[2]
ENV_CFG = REPO_PATH / "examples/embodiment/config/env/opensora_libero_spatial.yaml"
DEFAULT_MODEL_CFG = REPO_PATH / "examples/embodiment/config/model/multitask_dit.yaml"
CHUNK = 8
RECORD_KEYS = (
    "state_id",
    "task_id",
    "trial_id",
    "rerun",
    "task",
    "anchor_step",
    "source_rerun",
    "success",
    "first_success_step",
    "success_at_end",
    "max_rm_reward",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", required=True, help="collect_rollouts.py output.")
    parser.add_argument("--wm-ckpt", required=True, help="OpenSora world-model dir.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--mode", choices=["policy", "replay"], default="policy")
    parser.add_argument("--ckpt", help="Policy full_weights.pt (policy mode).")
    parser.add_argument("--model-cfg", default=str(DEFAULT_MODEL_CFG))
    parser.add_argument(
        "--trials",
        default=None,
        help="Trial ids per task, as 'a-b' (inclusive) or a comma list; "
        "default: every trial in --real.",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="Task ids, as 'a-b' (inclusive) or a comma list; default: every task.",
    )
    parser.add_argument(
        "--reruns",
        type=int,
        default=1,
        help="Rollouts per state in policy mode; real reruns replayed in replay mode.",
    )
    parser.add_argument(
        "--anchor-step",
        type=int,
        default=0,
        help=f"Real step the episode starts from; a multiple of {CHUNK}.",
    )
    parser.add_argument(
        "--anchors",
        default=None,
        help="anchor/select.py output with per-state start points (policy mode); "
        "replaces --anchor-step.",
    )
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--max-episode-steps", type=int, default=240)
    parser.add_argument("--success-threshold", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-fused-kernels",
        action="store_true",
        help="Run the world model with PyTorch attention and LayerNorm instead of "
        "flash-attn and apex, for GPUs their wheels have no kernels for (sm_120).",
    )
    return parser.parse_args()


def load_anchors(path: str, states: list[int]) -> dict[int, list[tuple[int, int]]]:
    """Each state's ``(real rerun, anchor step)`` start points from ``path``."""
    sources = json.loads(Path(path).read_text())["sources"]
    missing = [s for s in states if str(s) not in sources]
    if missing:
        raise ValueError(
            f"Expected start points for every state in {path}, found none for "
            f"states {missing}."
        )
    return {
        s: [(e["rerun"], e["anchor_step"]) for e in sources[str(s)]] for s in states
    }


def write_init_frames(
    real: Path, anchors: dict[tuple[int, int], int], out: Path
) -> dict[tuple[int, int], int]:
    """Write one single-frame trajectory per ``(state_id, rerun)`` source episode,
    from its frame at its anchor step.

    Returns each source's index in the world model's dataset, which lists the
    trajectory files in sorted order.
    """
    init_dir = out / "init_frames"
    init_dir.mkdir(parents=True, exist_ok=True)
    for stale in init_dir.glob("*.npy"):
        stale.unlink()
    for (state_id, rerun), anchor in anchors.items():
        name = episode_name(state_id, rerun)
        with np.load(real / "episodes" / f"{name}.npz") as episode:
            frame = {
                "image": episode["frames"][anchor],
                "reward": 0.0,
                "instruction": str(episode["task"]),
                "delta_action": np.zeros(7, dtype=np.float64),
            }
        np.save(
            init_dir / f"{name}.npy", np.array([frame], dtype=object), allow_pickle=True
        )
    names = sorted(str(p) for p in init_dir.glob("*.npy"))
    return {
        src: names.index(str(init_dir / f"{episode_name(*src)}.npy")) for src in anchors
    }


def build_env(args: argparse.Namespace, init_dir: Path, num_steps: int):
    from rlinf.envs.sim.world_model import WorldModelEnv

    cfg = OmegaConf.load(ENV_CFG)
    cfg.opensora_wm_hf_ckpt_path = str(Path(args.wm_ckpt).resolve())
    cfg.initial_image_path = str(init_dir)
    cfg.ignore_terminations = True
    cfg.auto_reset = False
    cfg.use_fixed_reset_state_ids = False
    cfg.use_rel_reward = False
    cfg.max_episode_steps = num_steps
    cfg.max_steps_per_rollout_epoch = num_steps
    cfg.success_reward_threshold = args.success_threshold
    cfg.seed = args.seed
    cfg.video_cfg.save_video = False
    if args.no_fused_kernels:
        cfg.world_model_cfg.model.enable_flash_attn = False
        cfg.world_model_cfg.model.enable_layernorm_kernel = False
    return WorldModelEnv(
        cfg, num_envs=args.num_envs, seed_offset=0, total_num_processes=1
    )


def to_uint8(frames: torch.Tensor) -> np.ndarray:
    """``[B, 3, T, H, W]`` in ``[-1, 1]`` to ``[B, T, H, W, 3]`` uint8, as the env does."""
    frames = ((frames.float() + 1.0) / 2.0 * 255.0).clamp(0, 255).to(torch.uint8)
    return frames.permute(0, 2, 3, 4, 1).cpu().numpy()


def main() -> None:
    args = parse_args()
    if args.mode == "policy" and args.ckpt is None:
        raise ValueError("Expected --ckpt in policy mode, got none.")
    if args.anchors and (args.mode != "policy" or args.anchor_step):
        raise ValueError(
            f"Expected --anchors in policy mode with --anchor-step 0, got mode "
            f"{args.mode} and --anchor-step {args.anchor_step}."
        )
    if args.max_episode_steps % CHUNK != 0:
        raise ValueError(
            f"Expected --max-episode-steps divisible by the world-model chunk {CHUNK}, "
            f"got {args.max_episode_steps}."
        )
    real, out = Path(args.real), Path(args.out)
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    trials = parse_ids(args.trials) if args.trials else None
    tasks = parse_ids(args.tasks) if args.tasks else None
    meta = {
        r["state_id"]: (r["task_id"], r["trial_id"], r["task"])
        for r in load_records(real)
        if (trials is None or r["trial_id"] in trials)
        and (tasks is None or r["task_id"] in tasks)
    }
    states = sorted(meta)
    jobs = [(s, k) for k in range(args.reruns) for s in states]
    # Each job's (real rerun, anchor step): with --anchors the state's start points
    # in turn, else policy episodes start from rerun 0 and replays from their own
    # episode.
    if args.anchors:
        points = load_anchors(args.anchors, states)
        start = {(s, k): points[s][k % len(points[s])] for s, k in jobs}
    else:
        start = {
            (s, k): (0 if args.mode == "policy" else k, args.anchor_step)
            for s, k in jobs
        }
    for job, (_, anchor) in start.items():
        if anchor % CHUNK or not 0 <= anchor < args.max_episode_steps:
            raise ValueError(
                f"Expected anchor steps that are multiples of {CHUNK} in [0, "
                f"{args.max_episode_steps}), got {anchor} for {episode_name(*job)}."
            )
    anchors = {(s, r): a for (s, _), (r, a) in start.items()}
    dataset_index = write_init_frames(real, anchors, out)
    num_episodes = len(jobs)
    steps = sorted({a for _, a in start.values()})
    logger.info(
        f"{args.mode}: {num_episodes} episodes from {len(states)} states x "
        f"{args.reruns} reruns from {len(anchors)} real episodes at steps "
        f"{steps[0]}-{steps[-1]}, {args.num_envs} envs per batch"
    )
    # Episodes a previous run of the same command saved are kept, not rerun.
    results = []
    for job in jobs:
        path = out / "episodes" / f"{episode_name(*job)}.npz"
        if path.exists():
            with np.load(path) as episode:
                record = {k: episode[k].item() for k in RECORD_KEYS}
            found = (record["source_rerun"], record["anchor_step"])
            if found != start[job]:
                raise ValueError(
                    f"Expected saved {path.name} to start from (rerun, step) "
                    f"{start[job]}, found {found}; use a new --out."
                )
            results.append(record)
    if results:
        done = {(r["state_id"], r["rerun"]) for r in results}
        jobs = [job for job in jobs if job not in done]
        logger.info(f"Kept {len(results)} saved episodes, {len(jobs)} to run")
    # Sorted by anchor step, a batch's episodes have about the same length.
    jobs.sort(key=lambda job: start[job][1])

    env = build_env(args, out / "init_frames", args.max_episode_steps)
    if env.chunk != CHUNK:
        raise ValueError(f"Expected a world-model chunk of {CHUNK}, got {env.chunk}.")
    policy = (
        load_policy(args.model_cfg, args.ckpt, args.max_episode_steps // CHUNK)
        if args.mode == "policy"
        else None
    )

    for batch_start in range(0, len(jobs), args.num_envs):
        batch = jobs[batch_start : batch_start + args.num_envs]
        # The env needs a full batch; padding slots repeat the first job and are dropped.
        padded = batch + [batch[0]] * (args.num_envs - len(batch))
        # Each episode ends at the real episode's last step; the batch runs until
        # its earliest-anchored episode does.
        num_chunks = (args.max_episode_steps - start[batch[0]][1]) // CHUNK
        if policy is None:
            real_actions = []
            for job in padded:
                rerun, anchor = start[job]
                name = episode_name(job[0], rerun)
                with np.load(real / "episodes" / f"{name}.npz") as ep:
                    real_actions.append(ep["actions"][anchor:])
            real_actions = np.stack(real_actions).astype(np.float32)
        obs, _ = env.reset(
            episode_indices=np.array(
                [dataset_index[(job[0], start[job][0])] for job in padded]
            )
        )
        if policy is not None:
            policy.reset()

        frames = [obs["main_images"].cpu().numpy()[:, None]]
        actions, rm_reward = [], []
        for c in range(num_chunks):
            if policy is not None:
                chunk_actions, _ = policy.predict_action_batch(obs)
                chunk_actions = chunk_actions.numpy()
            else:
                chunk_actions = real_actions[:, c * CHUNK : (c + 1) * CHUNK]
            obs_list, chunk_rewards, _, _, _ = env.chunk_step(chunk_actions)
            obs = obs_list[0]
            frames.append(to_uint8(env.current_obs[:, :, 0, -CHUNK:]))
            actions.append(chunk_actions)
            rm_reward.append(chunk_rewards.float().cpu().numpy())

        frames = np.concatenate(frames, axis=1)  # [B, T + 1, H, W, 3]
        actions = np.concatenate(actions, axis=1)  # [B, T, 7]
        rm_reward = np.concatenate(rm_reward, axis=1)  # [B, T]
        for slot, (state_id, rerun) in enumerate(batch):
            task_id, trial_id, task = meta[state_id]
            source_rerun, anchor = start[(state_id, rerun)]
            length = args.max_episode_steps - anchor
            step_success = rm_reward[slot, :length] >= args.success_threshold
            hits = np.flatnonzero(step_success)
            record = {
                "state_id": state_id,
                "task_id": task_id,
                "trial_id": trial_id,
                "rerun": rerun,
                "task": task,
                "anchor_step": anchor,
                "source_rerun": source_rerun,
                "success": bool(len(hits)),
                "first_success_step": int(hits[0]) + 1 if len(hits) else -1,
                "success_at_end": bool(step_success[-1]),
                "max_rm_reward": float(rm_reward[slot, :length].max()),
            }
            save_episode(
                out,
                record,
                frames=frames[slot, : length + 1],
                actions=actions[slot, :length],
                rm_reward=rm_reward[slot, :length],
                step_success=step_success,
            )
            results.append(record)
        rate = np.mean([r["success"] for r in results])
        logger.info(
            f"{len(results)}/{num_episodes} episodes, imagined success so far {rate:.3f}"
        )

    summary = write_summary(
        out,
        results,
        mode=args.mode,
        real=os.path.abspath(args.real),
        wm_ckpt=os.path.abspath(args.wm_ckpt),
        ckpt=os.path.abspath(args.ckpt) if args.ckpt else None,
        model_cfg=os.path.abspath(args.model_cfg),
        reruns=args.reruns,
        anchor_step=args.anchor_step,
        anchors=os.path.abspath(args.anchors) if args.anchors else None,
        success_threshold=args.success_threshold,
    )
    logger.info(
        f"Imagined success rate {summary['success_rate']:.3f} over {len(results)} "
        f"episodes (at step {args.max_episode_steps}: "
        f"{summary['success_at_end_rate']:.3f})"
    )


if __name__ == "__main__":
    main()
