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

"""Pick where imagined rollouts start: shortly before a real episode starts to fail.

``--real`` is a finished ``collect_rollouts.py`` run of the policy ``--ckpt``.
``--method`` (see ``score.py``) scores every chunk boundary of every episode;
the scores are cached in ``<method>.npz`` beside ``--out``.

A task's threshold at chunk ``c`` is the ``--quantile`` of the score over the task's
successful episodes, before their first success, at chunks within ``--window`` of
``c``; with fewer than ``--min-calibration`` values it pools every task. A failed
episode's alarm is the last chunk of its first run of ``--consecutive`` chunks above
the threshold, and its anchor is ``--lead`` steps earlier, at most ``--max-anchor``.
A failed episode without an alarm anchors ``--lead`` steps before its largest
excess over the threshold.

Writes ``--out`` (``anchors.json``): the method and settings, the thresholds per
task, one row per episode (scores, alarm, anchor), and ``sources``, each state's
start points for ``imagine_rollouts.py --anchors``: its failed episodes at their
anchors, or rerun 0 at step 0 when no rerun failed.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from rlinf.utils.logging import get_logger

from ..common import episode_name
from .score import SCORES, AnchorScore, RealRun

logger = get_logger()

REPO_PATH = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_CFG = REPO_PATH / "examples/embodiment/config/model/multitask_dit.yaml"
RECORD_FIELDS = ("state_id", "task_id", "rerun", "success", "first_success_step")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", required=True, help="collect_rollouts.py output.")
    parser.add_argument("--ckpt", required=True, help="The real run's policy weights.")
    parser.add_argument("--model-cfg", default=str(DEFAULT_MODEL_CFG))
    parser.add_argument("--out", required=True, help="anchors.json to write.")
    parser.add_argument("--method", required=True, choices=sorted(SCORES))
    parser.add_argument(
        "--batch-size", type=int, default=480, help="Sampled chunks per policy call."
    )
    parser.add_argument("--quantile", type=float, default=0.9)
    parser.add_argument(
        "--window", type=int, default=2, help="Chunks pooled on each side of c."
    )
    parser.add_argument("--min-calibration", type=int, default=20)
    parser.add_argument("--consecutive", type=int, default=2)
    parser.add_argument("--lead", type=int, default=16, help="Steps before the alarm.")
    parser.add_argument("--max-anchor", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    for name in sorted(SCORES):
        SCORES[name].add_args(parser.add_argument_group(name))
    return parser.parse_args()


def cached_scores(method: AnchorScore, run: RealRun, args: argparse.Namespace):
    """``method``'s scores of ``run``, from ``<method>.npz`` beside ``--out`` or scored."""
    cache = Path(args.out).parent / f"{method.name}.npz"
    meta = {
        "ckpt": os.path.abspath(args.ckpt),
        "seed": args.seed,
        "settings": method.settings(),
    }
    if cache.exists():
        with np.load(cache) as cached:
            found = json.loads(str(cached["meta"]))
            if found != meta or list(cached["names"]) != run.names:
                raise ValueError(
                    f"Expected {cache} scored with {meta} on the episodes of "
                    f"{run.path}, found {found}; delete it to rescore."
                )
            logger.info(f"Loaded {method.name} scores from {cache}")
            return cached["scores"]
    scores = method.score(run)
    expected = (len(run.records), run.num_chunks)
    if scores.shape != expected:
        raise ValueError(
            f"Expected {method.name} scores of shape {expected}, got {scores.shape}."
        )
    np.savez(cache, names=np.array(run.names), scores=scores, meta=json.dumps(meta))
    logger.info(f"Wrote {cache}")
    return scores


def thresholds(
    scores: np.ndarray, run: RealRun, args: argparse.Namespace
) -> dict[int, np.ndarray]:
    """Per task, the threshold at each chunk from its successful episodes."""
    num_chunks = run.num_chunks
    calib = defaultdict(lambda: [[] for _ in range(num_chunks)])
    for row, record in zip(scores, run.records):
        if record["success"]:
            for c in range(run.pre_success_chunks(record)):
                calib[record["task_id"]][c].append(row[c])
                calib["all"][c].append(row[c])
    if not calib["all"][0]:
        raise ValueError(f"Expected successful episodes in {run.path}, found none.")

    def window(values: list[list[float]], c: int) -> list[float]:
        lo, hi = max(0, c - args.window), min(num_chunks, c + args.window + 1)
        return [v for vs in values[lo:hi] for v in vs]

    pooled_all = [v for vs in calib["all"] for v in vs]
    taus, pooled, everything = {}, 0, 0
    for task in sorted({r["task_id"] for r in run.records}):
        tau = np.zeros(num_chunks)
        for c in range(num_chunks):
            values = window(calib[task], c) if task in calib else []
            if len(values) < args.min_calibration:
                values = window(calib["all"], c)
                pooled += 1
            if len(values) < args.min_calibration:
                values = pooled_all
                everything += 1
            tau[c] = np.quantile(values, args.quantile)
        taus[task] = tau
    if pooled:
        logger.info(
            f"{pooled} task-chunk thresholds had under {args.min_calibration} values "
            f"and pooled every task; {everything} of them pooled every chunk too"
        )
    return taus


def first_alarm(above: np.ndarray, consecutive: int) -> int:
    """Last chunk of the first run of ``consecutive`` chunks above threshold, or -1."""
    run = 0
    for c, hit in enumerate(above):
        run = run + 1 if hit else 0
        if run == consecutive:
            return c
    return -1


def select(
    scores: np.ndarray, run: RealRun, args: argparse.Namespace
) -> tuple[dict, list[dict], dict]:
    """Thresholds, one row per episode, and each state's start points."""
    taus = thresholds(scores, run, args)
    chunk = run.chunk
    rows, sources = [], defaultdict(list)
    for row, record in zip(scores, run.records):
        tau = taus[record["task_id"]]
        alarm = first_alarm(row > tau, args.consecutive)
        entry = {
            "episode": episode_name(record["state_id"], record["rerun"]),
            **{k: record[k] for k in RECORD_FIELDS},
            "scores": [round(float(v), 5) for v in row],
            "alarm_step": alarm * chunk if alarm >= 0 else -1,
            "anchor_step": None,
            "anchor_reason": None,
        }
        if not record["success"]:
            if alarm >= 0:
                anchor_chunk, reason = alarm, "alarm"
            else:
                anchor_chunk, reason = int(np.argmax(row - tau)), "peak"
            anchor = int(np.clip(anchor_chunk * chunk - args.lead, 0, args.max_anchor))
            entry["anchor_step"], entry["anchor_reason"] = anchor, reason
            sources[record["state_id"]].append(
                {"rerun": record["rerun"], "anchor_step": anchor}
            )
        rows.append(entry)
    for record in run.records:
        if record["state_id"] not in sources:
            sources[record["state_id"]] = [{"rerun": 0, "anchor_step": 0}]
    return taus, rows, sources


def log_summary(name: str, rows: list[dict], sources: dict, out: Path) -> None:
    failed = [r for r in rows if not r["success"]]
    succeeded = [r for r in rows if r["success"]]
    alarmed = [r for r in failed if r["anchor_reason"] == "alarm"]
    false_alarms = [
        r for r in succeeded if 0 <= r["alarm_step"] < r["first_success_step"]
    ]
    logger.info(
        f"{name}: {len(failed)} failed episodes, {len(alarmed)} with an alarm; "
        f"{len(false_alarms)}/{len(succeeded)} successes alarmed before success"
    )
    anchors = [r["anchor_step"] for r in failed]
    if anchors:
        logger.info(
            f"Anchor steps of failed episodes: median {np.median(anchors):.0f}, "
            f"quartiles {np.percentile(anchors, 25):.0f}-{np.percentile(anchors, 75):.0f}"
        )
    anchored = len({r["state_id"] for r in failed})
    logger.info(f"{len(sources)} states, {anchored} with failure anchors; wrote {out}")


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    chunk = OmegaConf.load(args.model_cfg).num_action_chunks
    for name in ("lead", "max_anchor"):
        if getattr(args, name) % chunk:
            raise ValueError(
                f"Expected --{name.replace('_', '-')} a multiple of the chunk {chunk}, "
                f"got {getattr(args, name)}."
            )
    run = RealRun(args.real, chunk)
    method = SCORES[args.method](args)
    scores = cached_scores(method, run, args)
    taus, rows, sources = select(scores, run, args)

    settings = {
        k: getattr(args, k)
        for k in (
            "quantile",
            "window",
            "min_calibration",
            "consecutive",
            "lead",
            "max_anchor",
            "seed",
        )
    }
    out.write_text(
        json.dumps(
            {
                "method": method.name,
                "real": os.path.abspath(args.real),
                "ckpt": os.path.abspath(args.ckpt),
                "chunk": chunk,
                "settings": {**settings, method.name: method.settings()},
                "thresholds": {
                    str(t): [round(float(v), 5) for v in tau] for t, tau in taus.items()
                },
                "sources": {str(s): sources[s] for s in sorted(sources)},
                "episodes": rows,
            },
            indent=1,
        )
    )
    log_summary(method.name, rows, sources, out)


if __name__ == "__main__":
    main()
