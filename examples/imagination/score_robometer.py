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

"""Score episodes from ``collect_rollouts.py`` with Robometer and compare to the simulator.

Runs in an environment with ``lerobot[robometer]`` installed. Each episode ends at
its first simulator success (or at its last step on failure). Robometer sees
``--num-frames`` frames spread evenly from the reset frame to a cut-off step and
returns per-frame progress and success probability. Cut-offs are the episode end
plus offsets around it (``--offsets``), so the success head can be checked
before and after the true success step.

Writes ``robometer_scores_<episode-end>.jsonl`` (one row per episode and cut-off)
and ``robometer_summary_<episode-end>.json`` next to the episodes.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

# Runs in the Robometer environment, which does not install RLinf's dependencies.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", required=True, help="collect_rollouts.py output.")
    parser.add_argument("--model", default="lerobot/Robometer-4B")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument(
        "--offsets",
        default="-48,-24,0,24,48",
        help="Cut-offs relative to the episode end, in env steps.",
    )
    parser.add_argument(
        "--episode-end",
        choices=["first_success", "last_step"],
        default="first_success",
        help="Where an episode ends: at its first success, as with "
        "ignore_terminations: False, or at its last step, as with True.",
    )
    parser.add_argument(
        "--lags",
        default="0,8,16,24,32,48",
        help="Frame-level check: compare each frame's score with the simulator's "
        "success this many steps earlier.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Probability that a random positive outscores a random negative."""
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return float(greater + 0.5 * ties)


def main() -> None:
    args = parse_args()
    from lerobot.rewards.robometer import RobometerConfig, RobometerRewardModel
    from lerobot.rewards.robometer.modeling_robometer import decode_progress_outputs
    from lerobot.rewards.robometer.processor_robometer import (
        RobometerEncoderProcessorStep,
    )

    cfg = RobometerConfig(pretrained_path=args.model, device="cuda")
    model = RobometerRewardModel.from_pretrained(args.model, config=cfg).to("cuda")
    model.eval()
    encoder = RobometerEncoderProcessorStep(
        base_model_id=cfg.base_model_id,
        use_multi_image=cfg.use_multi_image,
        use_per_frame_progress_token=cfg.use_per_frame_progress_token,
        max_frames=None,
    )
    offsets = [int(x) for x in args.offsets.split(",")]

    rollouts = Path(args.rollouts)
    queries = []
    for path in sorted((rollouts / "episodes").glob("*.npz")):
        episode = np.load(path)
        num_steps = len(episode["step_success"])
        step_success = episode["step_success"]
        success = bool(episode["success"])
        if args.episode_end == "first_success" and success:
            end, episode_offsets = int(episode["first_success_step"]), offsets
        else:
            end, episode_offsets = num_steps, [0]
        for offset in episode_offsets:
            cut = end + offset
            if cut < args.num_frames or cut > num_steps:
                continue
            frame_ids = np.linspace(0, cut, args.num_frames).round().astype(int)
            queries.append(
                {
                    "episode": path.stem,
                    "task": str(episode["task"]),
                    "sim_success": success,
                    "episode_end": end,
                    "offset": offset,
                    "cut": cut,
                    # The simulator's goal check at the cut-off step itself.
                    "sim_success_at_cut": bool(step_success[cut - 1]),
                    "frame_ids": frame_ids.tolist(),
                    # Simulator success after step t is step_success[t - 1].
                    "sim_success_by_step": step_success.astype(bool).tolist(),
                    "path": path,
                }
            )
    logger.info(f"{len(queries)} Robometer queries from {rollouts}")

    rows = []
    for start in range(0, len(queries), args.batch_size):
        batch = queries[start : start + args.batch_size]
        samples = [
            (np.load(q["path"])["frames"][q["frame_ids"]], q["task"]) for q in batch
        ]
        inputs = encoder.encode_samples(samples)
        inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            progress_logits, success_logits = model._compute_rbm_logits(inputs)
        decoded = decode_progress_outputs(
            progress_logits, success_logits, is_discrete_mode=cfg.use_discrete_progress
        )
        for q, progress, success_prob in zip(
            batch, decoded["progress_pred"], decoded["success_probs"]
        ):
            row = {k: v for k, v in q.items() if k != "path"}
            row["progress"] = [float(np.clip(p, 0.0, 1.0)) for p in progress]
            row["success_prob"] = [float(p) for p in success_prob]
            rows.append(row)
        logger.info(f"{len(rows)}/{len(queries)} queries scored")

    with open(rollouts / f"robometer_scores_{args.episode_end}.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    # Episode-level check at the episode end (offset 0), labeled by the simulator
    # at that step; with --episode-end last_step this is success at the last step.
    at_end = [r for r in rows if r["offset"] == 0]
    labels = np.array([r["sim_success_at_cut"] for r in at_end])
    success_once = np.array([r["sim_success"] for r in at_end])

    # Frame-level check: each scored frame against the simulator `lag` steps earlier.
    # A score that lags the true success state agrees best at a positive lag.
    by_lag = {}
    for lag in (int(x) for x in args.lags.split(",")):
        frame_scores, frame_progress, frame_labels = [], [], []
        for r in at_end:
            for i, frame in enumerate(r["frame_ids"]):
                step = frame - lag
                frame_scores.append(r["success_prob"][i])
                frame_progress.append(r["progress"][i])
                frame_labels.append(step >= 1 and r["sim_success_by_step"][step - 1])
        frame_labels = np.array(frame_labels)
        by_lag[lag] = {
            "frames": len(frame_labels),
            "sim_success_frames": int(frame_labels.sum()),
            "auroc_success_prob": auroc(np.array(frame_scores), frame_labels),
            "auroc_progress": auroc(np.array(frame_progress), frame_labels),
        }
    final_progress = np.array([r["progress"][-1] for r in at_end])
    final_success = np.array([r["success_prob"][-1] for r in at_end])
    predicted = final_success > cfg.success_threshold
    # Within successful episodes, progress should rise with time towards the end.
    time_corr = [
        float(np.corrcoef(np.array(r["frame_ids"]) / r["cut"], r["progress"])[0, 1])
        for r in at_end
        if r["sim_success"] and np.std(r["progress"]) > 0
    ]
    by_offset = {}
    for offset in offsets:
        sel = [r for r in rows if r["sim_success"] and r["offset"] == offset]
        if sel:
            by_offset[offset] = {
                "queries": len(sel),
                "sim_success_at_cut": float(
                    np.mean([r["sim_success_at_cut"] for r in sel])
                ),
                "mean_final_progress": float(np.mean([r["progress"][-1] for r in sel])),
                "mean_final_success_prob": float(
                    np.mean([r["success_prob"][-1] for r in sel])
                ),
                "predicted_success_rate": float(
                    np.mean(
                        [r["success_prob"][-1] > cfg.success_threshold for r in sel]
                    )
                ),
            }
    summary = {
        "model": args.model,
        "num_frames": args.num_frames,
        "episode_end": args.episode_end,
        "episodes": len(at_end),
        "sim_successes": int(labels.sum()),
        "auroc_final_progress": auroc(final_progress, labels),
        "auroc_final_success_prob": auroc(final_success, labels),
        "sim_success_once": int(success_once.sum()),
        "auroc_final_success_prob_vs_success_once": auroc(final_success, success_once),
        "frame_level_by_lag": by_lag,
        "success_head_threshold": cfg.success_threshold,
        "true_positive": int((predicted & labels).sum()),
        "false_positive": int((predicted & ~labels).sum()),
        "false_negative": int((~predicted & labels).sum()),
        "true_negative": int((~predicted & ~labels).sum()),
        "mean_final_progress_success": float(final_progress[labels].mean())
        if labels.any()
        else float("nan"),
        "mean_final_progress_failure": float(final_progress[~labels].mean())
        if (~labels).any()
        else float("nan"),
        "progress_time_corr_success_median": float(np.median(time_corr))
        if time_corr
        else float("nan"),
        "success_episodes_by_cut_offset": by_offset,
    }
    (rollouts / f"robometer_summary_{args.episode_end}.json").write_text(
        json.dumps(summary, indent=2)
    )
    logger.info(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
