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

"""Label episodes as successes with Robometer alone, real or imagined.

Runs in an environment with ``lerobot[robometer]`` installed. Each episode in
``<rollouts>/episodes/*.npz`` is scored in one Robometer query over its frames at
steps 0, ``--stride``, ..., the last step, with the episode's instruction. An episode
succeeds when ``--consecutive`` scored frames in a row have a success probability
above ``--threshold``; ``success_step`` is the step of the first frame of that window.
``mean_score`` is the mean success probability over the scored frames after the
reset frame, used to rank successes within a state.

Writes ``robometer_judge.jsonl`` (one row per episode) and
``robometer_judge_summary.json`` (success rates overall and per task) to
``<rollouts>``.
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# Runs in the Robometer environment, which does not install RLinf's dependencies.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

RECORD_KEYS = ("state_id", "task_id", "trial_id", "rerun", "task")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rollouts", required=True, help="Directory with episodes/*.npz."
    )
    parser.add_argument("--model", default="lerobot/Robometer-4B")
    parser.add_argument(
        "--stride", type=int, default=8, help="Env steps between frames."
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--consecutive", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def success_window(probs: np.ndarray, threshold: float, consecutive: int) -> int:
    """Index of the first frame of the first run of ``consecutive`` scores above threshold, or -1."""
    run = 0
    for i, p in enumerate(probs):
        run = run + 1 if p > threshold else 0
        if run == consecutive:
            return i - consecutive + 1
    return -1


def main() -> None:
    args = parse_args()
    from lerobot.rewards.robometer import RobometerConfig, RobometerRewardModel
    from lerobot.rewards.robometer.modeling_robometer import decode_progress_outputs
    from lerobot.rewards.robometer.processor_robometer import (
        RobometerEncoderProcessorStep,
    )

    cfg = RobometerConfig(pretrained_path=args.model, device=args.device)
    model = RobometerRewardModel.from_pretrained(args.model, config=cfg).to(args.device)
    model.eval()
    encoder = RobometerEncoderProcessorStep(
        base_model_id=cfg.base_model_id,
        use_multi_image=cfg.use_multi_image,
        use_per_frame_progress_token=cfg.use_per_frame_progress_token,
        max_frames=None,
    )

    rollouts = Path(args.rollouts)
    paths = sorted((rollouts / "episodes").glob("*.npz"))
    logger.info(f"Judging {len(paths)} episodes in {rollouts}")
    # A Robometer query batches episodes of one length only; imagined episodes from
    # failure anchors are shorter than the others.
    by_length = defaultdict(list)
    for path in paths:
        with np.load(path) as episode:
            by_length[len(episode["actions"])].append(path)
    batches = [
        group[start : start + args.batch_size]
        for _, group in sorted(by_length.items())
        for start in range(0, len(group), args.batch_size)
    ]
    rows = []
    for batch_paths in batches:
        batch, samples = [], []
        for path in batch_paths:
            with np.load(path) as episode:
                num_frames = len(episode["frames"])
                if (num_frames - 1) % args.stride != 0:
                    raise ValueError(
                        f"Expected {path.name} to have a multiple of --stride={args.stride} "
                        f"steps, got {num_frames - 1}."
                    )
                steps = list(range(0, num_frames, args.stride))
                record = {k: episode[k].item() for k in RECORD_KEYS}
                samples.append((episode["frames"][steps], str(record["task"])))
            batch.append({"episode": path.stem, **record, "steps": steps})
        inputs = encoder.encode_samples(samples)
        inputs = {
            k: v.to(args.device) if hasattr(v, "to") else v for k, v in inputs.items()
        }
        with torch.no_grad():
            progress_logits, success_logits = model._compute_rbm_logits(inputs)
        decoded = decode_progress_outputs(
            progress_logits, success_logits, is_discrete_mode=cfg.use_discrete_progress
        )
        for row, progress, success_prob in zip(
            batch, decoded["progress_pred"], decoded["success_probs"]
        ):
            probs = np.asarray(success_prob, dtype=np.float64)
            # The reset frame never shows a success, so the window starts after it.
            window = success_window(probs[1:], args.threshold, args.consecutive)
            row["success_prob"] = [float(p) for p in probs]
            row["progress"] = [float(np.clip(p, 0.0, 1.0)) for p in progress]
            row["success"] = window >= 0
            row["success_step"] = row["steps"][window + 1] if window >= 0 else -1
            row["mean_score"] = float(probs[1:].mean())
            rows.append(row)
        logger.info(f"{len(rows)}/{len(paths)} episodes judged")

    rows.sort(key=lambda row: row["episode"])
    with open(rollouts / "robometer_judge.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row["success"])
    summary = {
        "model": args.model,
        "stride": args.stride,
        "threshold": args.threshold,
        "consecutive": args.consecutive,
        "episodes": len(rows),
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "per_task": {
            task: {"episodes": len(v), "success_rate": float(np.mean(v))}
            for task, v in sorted(by_task.items())
        },
    }
    (rollouts / "robometer_judge_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    logger.info(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
