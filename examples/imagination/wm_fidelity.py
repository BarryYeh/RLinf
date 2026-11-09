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

"""Compare world-model rollouts from ``imagine_rollouts.py`` with real LIBERO rollouts.

Every success call on world-model frames is the Robometer judge's
(``robometer_judge.jsonl`` in the world-model run); success means success at any
step.

``--policy`` (closed loop) is compared per state: the judged imagined success rate
against the simulator's real one, as Pearson and Spearman correlation, mean absolute
error, and MMRV (mean maximum rank violation). ``--replay`` (open loop on the real
actions) is compared per episode, with the judge also run on ``--real``: a replay
disagrees when the judge's calls on the real and the replayed frames differ, a fake
success when only the replay passes and a missed success when only the real one
does. It also reports the PSNR of replayed frames against the real ones over time.
Writes ``--out`` as JSON.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from common import load_records

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


def judge_rows(run: Path) -> dict[str, dict]:
    path = run / "robometer_judge.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Expected robometer_judge.py output {path}.")
    rows = (json.loads(line) for line in open(path))
    return {row["episode"]: row for row in rows}


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


def per_state_rates(rows) -> dict[int, float]:
    by_state = defaultdict(list)
    for r in rows:
        by_state[r["state_id"]].append(r["success"])
    return {s: float(np.mean(v)) for s, v in by_state.items()}


def closed_loop(real: Path, policy: Path) -> dict:
    real_rows = load_records(real)
    real_rate = per_state_rates(real_rows)
    imagined_rate = per_state_rates(judge_rows(policy).values())
    states = sorted(set(real_rate) & set(imagined_rate))
    real_sr = np.array([real_rate[s] for s in states])
    imagined_sr = np.array([imagined_rate[s] for s in states])
    task_of = {r["state_id"]: r["task_id"] for r in real_rows}
    per_task = {}
    for task in sorted({task_of[s] for s in states}):
        sel = np.array([task_of[s] == task for s in states])
        per_task[task] = {
            "states": int(sel.sum()),
            "real_success_rate": float(real_sr[sel].mean()),
            "imagined_success_rate": float(imagined_sr[sel].mean()),
            "pearson_states": pearson(real_sr[sel], imagined_sr[sel]),
        }
    task_real = np.array([v["real_success_rate"] for v in per_task.values()])
    task_imagined = np.array([v["imagined_success_rate"] for v in per_task.values()])
    return {
        "states": len(states),
        "real_success_rate": float(real_sr.mean()),
        "imagined_success_rate": float(imagined_sr.mean()),
        "pearson_states": pearson(real_sr, imagined_sr),
        "spearman_states": pearson(ranks(real_sr), ranks(imagined_sr)),
        "mae_states": float(np.abs(real_sr - imagined_sr).mean()),
        "mmrv_states": mmrv(real_sr, imagined_sr),
        "pearson_tasks": pearson(task_real, task_imagined),
        "mmrv_tasks": mmrv(task_real, task_imagined),
        "per_task": per_task,
    }


def psnr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-frame PSNR of uint8 ``[T, H, W, 3]`` arrays."""
    mse = ((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean(axis=(1, 2, 3))
    return 10 * np.log10(255.0**2 / np.maximum(mse, 1e-10))


def disagreement(sim: np.ndarray, real: np.ndarray, replay: np.ndarray) -> dict:
    return {
        "episodes": len(sim),
        "sim_success_rate": float(sim.mean()),
        "real_judge_success_rate": float(real.mean()),
        "replay_judge_success_rate": float(replay.mean()),
        "disagreement": float((real != replay).mean()),
        "fake_success": float((replay & ~real).mean()),
        "missed_success": float((~replay & real).mean()),
    }


def open_loop(real: Path, replay: Path, psnr_steps: list[int]) -> dict:
    real_judge, replay_judge = judge_rows(real), judge_rows(replay)
    names = sorted(replay_judge)
    task, sim, curves = [], [], []
    for name in names:
        with (
            np.load(replay / "episodes" / f"{name}.npz") as replayed,
            np.load(real / "episodes" / f"{name}.npz") as real_ep,
        ):
            anchor = int(replayed["anchor_step"]) if "anchor_step" in replayed else 0
            task.append(int(real_ep["task_id"]))
            sim.append(bool(real_ep["success"]))
            curves.append(psnr(replayed["frames"], real_ep["frames"][anchor:]))
    task, sim = np.array(task), np.array(sim)
    real_call = np.array([real_judge[n]["success"] for n in names])
    replay_call = np.array([replay_judge[n]["success"] for n in names])
    curves = np.stack(curves)
    return {
        **disagreement(sim, real_call, replay_call),
        "per_task": {
            int(t): disagreement(
                sim[task == t], real_call[task == t], replay_call[task == t]
            )
            for t in np.unique(task)
        },
        "psnr_by_step": {
            s: float(curves[:, s].mean()) for s in psnr_steps if s < curves.shape[1]
        },
    }


def main() -> None:
    args = parse_args()
    real = Path(args.real)
    report = {"real": str(real)}
    if args.policy:
        report["closed_loop"] = closed_loop(real, Path(args.policy))
    if args.replay:
        steps = [int(x) for x in args.psnr_steps.split(",")]
        report["open_loop"] = open_loop(real, Path(args.replay), steps)
    Path(args.out).write_text(json.dumps(report, indent=2))
    logger.info(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
