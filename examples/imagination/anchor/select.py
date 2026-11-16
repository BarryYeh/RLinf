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

``--real`` is a finished ``collect_rollouts.py`` run of the policy ``--ckpt``. Each
``--method`` (see ``score.py``) scores every chunk boundary of every episode; the
scores are cached in ``<method>.npz`` in ``--cache-dir``.

As in FIPER, each method's scores are summed over the last ``--agg-window`` chunks,
and a threshold is calibrated on the sums of the successful episodes before their
first success, at ``--quantile``: per task and chunk,
pooled over ``--pool`` chunks on each side (``--threshold step``), or per task over
each success's largest sum (``max``). A task with fewer than ``--min-calibration``
values pools every task; chunks past the last one with that many values keep its
threshold. A method's ratio is its sum over its threshold, so 1 is at threshold for
every method; ``--combine and`` takes the smallest ratio of the methods, ``or`` the
largest.

A failed episode's alarm is the first chunk where the ratio exceeds 1, and its anchor
is ``--lead`` steps earlier, at most ``--max-anchor``. A failed episode without an alarm anchors ``--lead`` steps before
its largest ratio.

Writes ``--out`` (``anchors.json``): the methods and settings, the thresholds per
method and task, one row per episode (raw scores, ratio, alarm, anchor), and
``sources``, each state's start points for ``imagine_rollouts.py --anchors``: its
failed episodes at their anchors, or rerun 0 at step 0 when no rerun failed.
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
SETTINGS = (
    "agg_window",
    "threshold",
    "quantile",
    "pool",
    "min_calibration",
    "combine",
    "lead",
    "max_anchor",
    "seed",
)
# FIPER's clip_threshold: a threshold of 0 (ACE where every success's samples agree)
# still alarms on any positive sum.
MIN_THRESHOLD = 1e-6


def positive_int(value: str) -> int:
    if not value.isdigit() or int(value) < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", required=True, help="collect_rollouts.py output.")
    parser.add_argument("--ckpt", required=True, help="The real run's policy weights.")
    parser.add_argument("--model-cfg", default=str(DEFAULT_MODEL_CFG))
    parser.add_argument("--out", required=True, help="anchors.json to write.")
    parser.add_argument(
        "--cache-dir", help="Where <method>.npz score caches live; default: --out's."
    )
    parser.add_argument("--method", required=True, nargs="+", choices=sorted(SCORES))
    parser.add_argument("--combine", choices=("and", "or"), default="and")
    parser.add_argument(
        "--batch-size", type=int, default=480, help="Sampled chunks per policy call."
    )
    parser.add_argument(
        "--agg-window", type=positive_int, default=6, help="Chunks summed."
    )
    parser.add_argument("--threshold", choices=("step", "max"), default="step")
    parser.add_argument("--quantile", type=float, default=0.9)
    parser.add_argument(
        "--pool", type=int, default=2, help="Chunks pooled on each side (step)."
    )
    parser.add_argument("--min-calibration", type=int, default=20)
    parser.add_argument("--lead", type=int, default=16, help="Steps before the alarm.")
    parser.add_argument("--max-anchor", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    for name in sorted(SCORES):
        SCORES[name].add_args(parser.add_argument_group(name))
    return parser.parse_args()


def cached_scores(method: AnchorScore, run: RealRun, args: argparse.Namespace):
    """``method``'s scores of ``run``, from ``<method>.npz`` in the cache dir or scored."""
    cache = Path(args.cache_dir or Path(args.out).parent) / f"{method.name}.npz"
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
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, names=np.array(run.names), scores=scores, meta=json.dumps(meta))
    logger.info(f"Wrote {cache}")
    return scores


def aggregate(scores: np.ndarray, window: int) -> np.ndarray:
    """Each chunk's score summed over the last ``window`` chunks."""
    totals = np.cumsum(np.pad(scores, ((0, 0), (1, 0))), axis=1)
    starts = np.maximum(np.arange(1, scores.shape[1] + 1) - window, 0)
    return totals[:, 1:] - totals[:, starts]


def thresholds(
    sums: np.ndarray, run: RealRun, args: argparse.Namespace
) -> dict[int, np.ndarray]:
    """Per task, the threshold at each chunk from its successful episodes' sums."""
    num_chunks = run.num_chunks
    tasks = sorted({r["task_id"] for r in run.records})
    calib = defaultdict(lambda: [[] for _ in range(num_chunks)])
    peaks = defaultdict(list)
    for row, record in zip(sums, run.records):
        before = row[: run.pre_success_chunks(record)] if record["success"] else []
        for c, value in enumerate(before):
            calib[record["task_id"]][c].append(value)
            calib["all"][c].append(value)
        if len(before):
            peaks[record["task_id"]].append(before.max())
            peaks["all"].append(before.max())
    if not peaks["all"]:
        raise ValueError(f"Expected successful episodes in {run.path}, found none.")

    if args.threshold == "max":
        taus = {}
        for task in tasks:
            values = peaks[task]
            if len(values) < args.min_calibration:
                values = peaks["all"]
            taus[task] = np.full(num_chunks, np.quantile(values, args.quantile))
        return taus

    def pooled(values: list[list[float]], c: int) -> list[float]:
        lo, hi = max(0, c - args.pool), min(num_chunks, c + args.pool + 1)
        return [v for vs in values[lo:hi] for v in vs]

    enough = [
        c
        for c in range(num_chunks)
        if len(pooled(calib["all"], c)) >= args.min_calibration
    ]
    if not enough:
        most = max(len(pooled(calib["all"], c)) for c in range(num_chunks))
        raise ValueError(
            f"Expected {args.min_calibration} success values at some chunk of "
            f"{run.path}, found at most {most}."
        )
    last = enough[-1]
    taus, pooled_tasks = {}, 0
    for task in tasks:
        tau = np.zeros(num_chunks)
        for c in range(last + 1):
            values = pooled(calib[task], c)
            if len(values) < args.min_calibration:
                values = pooled(calib["all"], c)
                pooled_tasks += 1
            tau[c] = np.quantile(values, args.quantile)
        tau[last + 1 :] = tau[last]
        taus[task] = tau
    logger.info(
        f"{pooled_tasks} task-chunk thresholds had under {args.min_calibration} "
        f"values and pooled every task; chunks after {last} keep chunk {last}'s"
    )
    return taus


def first_alarm(above: np.ndarray) -> int:
    """The first chunk where ``above`` holds, or -1."""
    hits = np.flatnonzero(above)
    return int(hits[0]) if len(hits) else -1


def select(
    scores: dict[str, np.ndarray], run: RealRun, args: argparse.Namespace
) -> tuple[dict, list[dict], dict]:
    """Thresholds per method, one row per episode, and each state's start points."""
    taus, ratios = {}, []
    for name, raw in scores.items():
        sums = aggregate(raw, args.agg_window)
        taus[name] = thresholds(sums, run, args)
        tau = np.stack([taus[name][r["task_id"]] for r in run.records])
        ratios.append(sums / np.maximum(tau, MIN_THRESHOLD))
    ratio = np.min(ratios, axis=0) if args.combine == "and" else np.max(ratios, axis=0)

    chunk = run.chunk
    rows, sources = [], defaultdict(list)
    for i, record in enumerate(run.records):
        alarm = first_alarm(ratio[i] > 1)
        entry = {
            "episode": episode_name(record["state_id"], record["rerun"]),
            **{k: record[k] for k in RECORD_FIELDS},
            "scores": {
                name: [round(float(v), 5) for v in raw[i]]
                for name, raw in scores.items()
            },
            "ratio": [round(float(v), 4) for v in ratio[i]],
            "alarm_step": alarm * chunk if alarm >= 0 else -1,
            "anchor_step": None,
            "anchor_reason": None,
        }
        if not record["success"]:
            if alarm >= 0:
                anchor_chunk, reason = alarm, "alarm"
            else:
                anchor_chunk, reason = int(np.argmax(ratio[i])), "peak"
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


def log_summary(
    label: str, rows: list[dict], sources: dict, out: Path, max_anchor: int
) -> None:
    failed = [r for r in rows if not r["success"]]
    succeeded = [r for r in rows if r["success"]]
    alarmed = [r for r in failed if r["anchor_reason"] == "alarm"]
    false_alarms = [
        r for r in succeeded if 0 <= r["alarm_step"] < r["first_success_step"]
    ]
    logger.info(
        f"{label}: {len(failed)} failed episodes, {len(alarmed)} with an alarm; "
        f"{len(false_alarms)}/{len(succeeded)} successes alarmed before success"
    )
    anchors = [r["anchor_step"] for r in failed]
    if anchors:
        logger.info(
            f"Anchor steps of failed episodes: median {np.median(anchors):.0f}, "
            f"quartiles {np.percentile(anchors, 25):.0f}-{np.percentile(anchors, 75):.0f}, "
            f"{np.mean(np.array(anchors) == max_anchor):.0%} at {max_anchor}"
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
    methods = [SCORES[name](args) for name in dict.fromkeys(args.method)]
    scores = {m.name: cached_scores(m, run, args) for m in methods}
    taus, rows, sources = select(scores, run, args)

    settings = {k: getattr(args, k) for k in SETTINGS}
    out.write_text(
        json.dumps(
            {
                "methods": list(scores),
                "real": os.path.abspath(args.real),
                "ckpt": os.path.abspath(args.ckpt),
                "chunk": chunk,
                "settings": {**settings, **{m.name: m.settings() for m in methods}},
                "thresholds": {
                    name: {
                        str(t): [round(float(v), 5) for v in tau]
                        for t, tau in method_taus.items()
                    }
                    for name, method_taus in taus.items()
                },
                "sources": {str(s): sources[s] for s in sorted(sources)},
                "episodes": rows,
            },
            indent=1,
        )
    )
    log_summary(f" {args.combine} ".join(scores), rows, sources, out, args.max_anchor)


if __name__ == "__main__":
    main()
