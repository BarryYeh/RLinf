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

"""Compare world-model rollouts from ``imagine_rollouts.py`` with the real LIBERO rollouts.

``--policy`` (closed loop) is compared per state: the imagined success rate against
the real one, as Pearson and Spearman correlation, mean absolute error, and MMRV
(mean maximum rank violation). ``--replay`` (open loop on the real actions) is
compared per episode: the reward model's success call on generated frames and on
the real frames against the simulator, and the PSNR of generated frames against the
real ones over time. Success means success at any step. Writes ``--out`` as JSON.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from rlinf.utils.logging import get_logger

logger = get_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", required=True, help="collect_rollouts.py output.")
    parser.add_argument("--policy", help="imagine_rollouts.py --mode policy output.")
    parser.add_argument("--replay", help="imagine_rollouts.py --mode replay output.")
    parser.add_argument("--out", required=True, help="Output JSON path.")
    parser.add_argument("--psnr-steps", default="8,24,48,96,144,192,240")
    return parser.parse_args()


def episodes(run: Path) -> list[dict]:
    return json.loads((run / "summary.json").read_text())["episodes_list"]


def ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks, ties sharing their mean rank."""
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for value in np.unique(x):
        tied = x == value
        r[tied] = r[tied].mean()
    return r


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def mmrv(real: np.ndarray, imagined: np.ndarray) -> float:
    """Mean over items of the largest real-rate gap whose order the imagined rates flip."""
    flipped = (real[:, None] < real[None, :]) != (imagined[:, None] < imagined[None, :])
    gaps = np.abs(real[:, None] - real[None, :]) * flipped
    return float(gaps.max(axis=1).mean())


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(
        (pos[:, None] > neg[None, :]).mean()
        + 0.5 * (pos[:, None] == neg[None, :]).mean()
    )


def confusion(pred: np.ndarray, label: np.ndarray) -> dict:
    return {
        "accuracy": float((pred == label).mean()),
        "true_positive": int((pred & label).sum()),
        "false_positive": int((pred & ~label).sum()),
        "false_negative": int((~pred & label).sum()),
        "true_negative": int((~pred & ~label).sum()),
    }


def per_state_rates(rows: list[dict]) -> dict[int, float]:
    by_state = defaultdict(list)
    for r in rows:
        by_state[r["state_id"]].append(r["success"])
    return {s: float(np.mean(v)) for s, v in by_state.items()}


def closed_loop(real_rows: list[dict], imagined_rows: list[dict]) -> dict:
    real_rate, imagined_rate = (
        per_state_rates(real_rows),
        per_state_rates(imagined_rows),
    )
    states = sorted(set(real_rate) & set(imagined_rate))
    real = np.array([real_rate[s] for s in states])
    imagined = np.array([imagined_rate[s] for s in states])
    task_of = {r["state_id"]: r["task_id"] for r in real_rows}
    per_task = {}
    for task in sorted({task_of[s] for s in states}):
        sel = np.array([task_of[s] == task for s in states])
        per_task[task] = {
            "states": int(sel.sum()),
            "real_success_rate": float(real[sel].mean()),
            "imagined_success_rate": float(imagined[sel].mean()),
            "pearson_states": pearson(real[sel], imagined[sel]),
        }
    task_real = np.array([v["real_success_rate"] for v in per_task.values()])
    task_imagined = np.array([v["imagined_success_rate"] for v in per_task.values()])
    return {
        "states": len(states),
        "real_success_rate": float(real.mean()),
        "imagined_success_rate": float(imagined.mean()),
        "pearson_states": pearson(real, imagined),
        "spearman_states": pearson(ranks(real), ranks(imagined)),
        "mae_states": float(np.abs(real - imagined).mean()),
        "mmrv_states": mmrv(real, imagined),
        "pearson_tasks": pearson(task_real, task_imagined),
        "mmrv_tasks": mmrv(task_real, task_imagined),
        "per_task": per_task,
    }


def psnr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-frame PSNR of uint8 ``[T, H, W, 3]`` arrays."""
    mse = ((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean(axis=(1, 2, 3))
    return 10 * np.log10(255.0**2 / np.maximum(mse, 1e-10))


def open_loop(real: Path, replay: Path, psnr_steps: list[int]) -> dict:
    sim, wm_call, real_call, wm_max, real_max, curves = [], [], [], [], [], []
    for r in episodes(replay):
        name = f"state_{r['state_id']:03d}_rerun_{r['rerun']}.npz"
        imagined = np.load(replay / "episodes" / name)
        real_ep = np.load(real / "episodes" / name)
        threshold_calls = imagined["step_success"]
        sim.append(bool(real_ep["success"]))
        wm_call.append(bool(threshold_calls.any()))
        wm_max.append(float(imagined["rm_reward"].max()))
        real_max.append(float(imagined["rm_reward_real"].max()))
        curves.append(psnr(imagined["frames"], real_ep["frames"]))
    sim, wm_call = np.array(sim), np.array(wm_call)
    wm_max, real_max = np.array(wm_max), np.array(real_max)
    threshold = json.loads((replay / "summary.json").read_text())["success_threshold"]
    real_call = real_max >= threshold
    curves = np.stack(curves)
    return {
        "episodes": len(sim),
        "sim_success_rate": float(sim.mean()),
        "wm_success_rate": float(wm_call.mean()),
        "rm_on_real_frames_success_rate": float(real_call.mean()),
        "wm_vs_sim": confusion(wm_call, sim),
        "rm_on_real_frames_vs_sim": confusion(real_call, sim),
        "auroc_wm_max_reward": auroc(wm_max, sim),
        "auroc_real_frames_max_reward": auroc(real_max, sim),
        "psnr_by_step": {
            s: float(curves[:, s].mean()) for s in psnr_steps if s < curves.shape[1]
        },
    }


def main() -> None:
    args = parse_args()
    real = Path(args.real)
    report = {"real": str(real)}
    if args.policy:
        report["closed_loop"] = closed_loop(episodes(real), episodes(Path(args.policy)))
    if args.replay:
        steps = [int(x) for x in args.psnr_steps.split(",")]
        report["open_loop"] = open_loop(real, Path(args.replay), steps)
    Path(args.out).write_text(json.dumps(report, indent=2))
    logger.info(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
