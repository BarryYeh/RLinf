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

"""Roll out episodes in the OpenSora world model from the initial frames of real LIBERO rollouts.

``--real`` is a ``collect_rollouts.py`` output. The reset frame of each state's first
rerun becomes that state's world-model initial frame, written to
``<out>/init_frames/state_<id>.npy`` in the world model's trajectory format.

``--mode policy`` runs the policy closed loop in the world model, ``--reruns`` times
per state. ``--mode replay`` feeds each real episode's actions to the world model open
loop, which measures the dynamics without the policy reacting to generated frames;
it also scores the real frames with the world model's reward model.

Each episode is written to ``<out>/episodes/state_<id>_rerun_<k>.npz`` with the
generated frames (the initial frame plus one per step), the actions, the reward
model's per-step success probability (``rm_reward``), its success call
(``step_success``: ``rm_reward >= --success-threshold``), and the LIBERO task, trial
and init-state ids; replay episodes also hold ``rm_reward_real``. ``summary.json``
holds the success rates. Run it in the OpenSora environment.
"""

import argparse
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.utils.logging import get_logger

logger = get_logger()

REPO_PATH = Path(__file__).resolve().parents[2]
ENV_CFG = REPO_PATH / "examples/embodiment/config/env/opensora_libero_spatial.yaml"
DEFAULT_MODEL_CFG = REPO_PATH / "examples/embodiment/config/model/multitask_dit.yaml"
RECORD_KEYS = (
    "state_id",
    "task_id",
    "trial_id",
    "rerun",
    "task",
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


def parse_trials(spec: str) -> list[int]:
    if "-" in spec:
        start, end = (int(x) for x in spec.split("-"))
        return list(range(start, end + 1))
    return [int(x) for x in spec.split(",")]


def load_real_records(real: Path) -> list[dict]:
    """Episode records of a finished run, or of the episodes a running one has saved."""
    if (real / "summary.json").exists():
        return json.loads((real / "summary.json").read_text())["episodes_list"]
    keys = ("state_id", "task_id", "trial_id", "rerun", "task", "success")
    records = []
    for path in sorted((real / "episodes").glob("*.npz")):
        try:
            with np.load(path) as episode:
                records.append({k: episode[k].item() for k in keys})
        except (EOFError, OSError, ValueError, zipfile.BadZipFile):
            # A running collection is still writing this episode.
            continue
    return records


def write_init_frames(real: Path, states: list[int], out: Path) -> Path:
    """One single-frame world-model trajectory per state, sorted by state id."""
    init_dir = out / "init_frames"
    init_dir.mkdir(parents=True, exist_ok=True)
    for state_id in states:
        episode = np.load(real / "episodes" / f"state_{state_id:03d}_rerun_0.npz")
        frame = {
            "image": episode["frames"][0],
            "reward": 0.0,
            "instruction": str(episode["task"]),
            "delta_action": np.zeros(7, dtype=np.float64),
        }
        np.save(
            init_dir / f"state_{state_id:03d}.npy",
            np.array([frame], dtype=object),
            allow_pickle=True,
        )
    return init_dir


def build_env(args: argparse.Namespace, init_dir: Path):
    from rlinf.envs.sim.world_model import WorldModelEnv

    cfg = OmegaConf.load(ENV_CFG)
    cfg.opensora_wm_hf_ckpt_path = str(Path(args.wm_ckpt).resolve())
    cfg.initial_image_path = str(init_dir)
    cfg.ignore_terminations = True
    cfg.auto_reset = False
    cfg.use_fixed_reset_state_ids = False
    cfg.use_rel_reward = False
    cfg.max_episode_steps = args.max_episode_steps
    cfg.max_steps_per_rollout_epoch = args.max_episode_steps
    cfg.success_reward_threshold = args.success_threshold
    cfg.seed = args.seed
    cfg.video_cfg.save_video = False
    if args.no_fused_kernels:
        cfg.world_model_cfg.model.enable_flash_attn = False
        cfg.world_model_cfg.model.enable_layernorm_kernel = False
    return WorldModelEnv(
        cfg, num_envs=args.num_envs, seed_offset=0, total_num_processes=1
    )


def build_policy(args: argparse.Namespace, num_chunks_per_episode: int):
    from rlinf.models import get_model

    cfg = OmegaConf.load(args.model_cfg)
    cfg.obs_history_reset_interval = num_chunks_per_episode
    policy = get_model(cfg)
    policy.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    return policy.to("cuda").eval()


def to_uint8(frames: torch.Tensor) -> np.ndarray:
    """``[B, 3, T, H, W]`` in ``[-1, 1]`` to ``[B, T, H, W, 3]`` uint8, as the env does."""
    frames = ((frames.float() + 1.0) / 2.0 * 255.0).clamp(0, 255).to(torch.uint8)
    return frames.permute(0, 2, 3, 4, 1).cpu().numpy()


@torch.no_grad()
def score_frames(env, frames: np.ndarray) -> np.ndarray:
    """Reward-model success probability of ``[N, H, W, 3]`` uint8 frames."""
    x = torch.from_numpy(frames).to(env.device).permute(0, 3, 1, 2).float()
    x = x / 255.0 * 2.0 - 1.0
    scores = [env.reward_model.predict_rew(chunk) for chunk in x.split(256)]
    return torch.cat(scores).float().cpu().numpy()


def main() -> None:
    args = parse_args()
    if args.mode == "policy" and args.ckpt is None:
        raise ValueError("Expected --ckpt in policy mode, got none.")
    real, out = Path(args.real), Path(args.out)
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    chunk = 8
    if args.max_episode_steps % chunk != 0:
        raise ValueError(
            f"Expected --max-episode-steps divisible by the world-model chunk {chunk}, "
            f"got {args.max_episode_steps}."
        )
    num_chunks = args.max_episode_steps // chunk

    records = load_real_records(real)
    trials = parse_trials(args.trials) if args.trials else None
    tasks = parse_trials(args.tasks) if args.tasks else None
    meta = {
        r["state_id"]: (r["task_id"], r["trial_id"], r["task"])
        for r in records
        if (trials is None or r["trial_id"] in trials)
        and (tasks is None or r["task_id"] in tasks)
    }
    states = sorted(meta)
    init_dir = write_init_frames(real, states, out)
    dataset_index = {state_id: i for i, state_id in enumerate(states)}
    jobs = [(s, k) for k in range(args.reruns) for s in states]
    num_episodes = len(jobs)
    logger.info(
        f"{args.mode}: {num_episodes} episodes from {len(states)} states x "
        f"{args.reruns} reruns, {args.num_envs} envs per batch"
    )
    # Episodes a previous run of the same command saved are kept, not rerun.
    results = []
    for state_id, rerun in jobs:
        path = out / "episodes" / f"state_{state_id:03d}_rerun_{rerun}.npz"
        if path.exists():
            with np.load(path) as episode:
                results.append({k: episode[k].item() for k in RECORD_KEYS})
    if results:
        done = {(r["state_id"], r["rerun"]) for r in results}
        jobs = [job for job in jobs if job not in done]
        logger.info(f"Kept {len(results)} saved episodes, {len(jobs)} to run")

    env = build_env(args, init_dir)
    if env.chunk != chunk:
        raise ValueError(f"Expected a world-model chunk of {chunk}, got {env.chunk}.")
    policy = build_policy(args, num_chunks) if args.mode == "policy" else None

    for batch_start in range(0, len(jobs), args.num_envs):
        batch = jobs[batch_start : batch_start + args.num_envs]
        # The env needs a full batch; padding slots repeat the first job and are dropped.
        padded = batch + [batch[0]] * (args.num_envs - len(batch))
        if args.mode == "replay":
            real_eps = [
                np.load(real / "episodes" / f"state_{s:03d}_rerun_{k}.npz")
                for s, k in padded
            ]
            real_actions = np.stack([ep["actions"] for ep in real_eps]).astype(
                np.float32
            )
        obs, _ = env.reset(
            episode_indices=np.array([dataset_index[s] for s, _ in padded])
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
                chunk_actions = real_actions[:, c * chunk : (c + 1) * chunk]
            obs_list, chunk_rewards, _, _, _ = env.chunk_step(chunk_actions)
            obs = obs_list[0]
            frames.append(to_uint8(env.current_obs[:, :, 0, -chunk:]))
            actions.append(chunk_actions)
            rm_reward.append(chunk_rewards.float().cpu().numpy())

        frames = np.concatenate(frames, axis=1)  # [B, T + 1, H, W, 3]
        actions = np.concatenate(actions, axis=1)  # [B, T, 7]
        rm_reward = np.concatenate(rm_reward, axis=1)  # [B, T]
        for slot, (state_id, rerun) in enumerate(batch):
            task_id, trial_id, task = meta[state_id]
            step_success = rm_reward[slot] >= args.success_threshold
            hits = np.flatnonzero(step_success)
            record = {
                "state_id": state_id,
                "task_id": task_id,
                "trial_id": trial_id,
                "rerun": rerun,
                "task": task,
                "success": bool(len(hits)),
                "first_success_step": int(hits[0]) + 1 if len(hits) else -1,
                "success_at_end": bool(step_success[-1]),
                "max_rm_reward": float(rm_reward[slot].max()),
            }
            extra = {}
            if args.mode == "replay":
                extra["rm_reward_real"] = score_frames(
                    env, real_eps[slot]["frames"][1:]
                )
                extra["real_step_success"] = real_eps[slot]["step_success"]
            np.savez_compressed(
                out / "episodes" / f"state_{state_id:03d}_rerun_{rerun}.npz",
                frames=frames[slot],
                actions=actions[slot],
                rm_reward=rm_reward[slot],
                step_success=step_success,
                **extra,
                **{k: np.asarray(v) for k, v in record.items()},
            )
            results.append(record)
        rate = np.mean([r["success"] for r in results])
        logger.info(
            f"{len(results)}/{num_episodes} episodes, imagined success so far {rate:.3f}"
        )

    per_task = {}
    for task_id in sorted({r["task_id"] for r in results}):
        rows = [r for r in results if r["task_id"] == task_id]
        per_task[task_id] = {
            "task": rows[0]["task"],
            "episodes": len(rows),
            "success_rate": float(np.mean([r["success"] for r in rows])),
        }
    summary = {
        "mode": args.mode,
        "real": os.path.abspath(args.real),
        "wm_ckpt": os.path.abspath(args.wm_ckpt),
        "ckpt": os.path.abspath(args.ckpt) if args.ckpt else None,
        "model_cfg": os.path.abspath(args.model_cfg),
        "reruns": args.reruns,
        "success_threshold": args.success_threshold,
        "episodes": len(results),
        "success_rate": float(np.mean([r["success"] for r in results])),
        "success_at_end_rate": float(np.mean([r["success_at_end"] for r in results])),
        "per_task": per_task,
        "episodes_list": results,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(
        f"Imagined success rate {summary['success_rate']:.3f} over {len(results)} "
        f"episodes (at step {args.max_episode_steps}: "
        f"{summary['success_at_end_rate']:.3f})"
    )


if __name__ == "__main__":
    main()
