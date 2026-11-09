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

"""Compare policy evaluations in the README's reporting format.

Each ``--arms`` entry is ``<name>=<collect_rollouts.py run>``, all on the same states
and reruns. The report leads with success at any step: the all-task table, then each
arm listed after ``--reference`` minus it, with a 95% paired bootstrap interval over states, then a
per-task table; the same follows for success at the last step. ``--selections``
adds the kept episodes and frames per task of ``export_imagined.py`` datasets, and
``--fidelity`` the open-loop table of a ``wm_fidelity.py`` report. Writes
``<out>.md`` and ``<out>.json``.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from common import load_records

from rlinf.utils.logging import get_logger

logger = get_logger()

METRICS = {"success": "Success at any step", "success_at_end": "Success at step 240"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", required=True, help="<name>=<run dir>")
    parser.add_argument("--reference", default="Demos only")
    parser.add_argument("--out", required=True, help="Output path without suffix.")
    parser.add_argument("--selections", nargs="*", default=[])
    parser.add_argument("--fidelity", default=None)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def state_rates(records: list[dict], metric: str) -> dict[int, float]:
    by_state = defaultdict(list)
    for r in records:
        by_state[r["state_id"]].append(r[metric])
    return {s: float(np.mean(v)) for s, v in sorted(by_state.items())}


def bootstrap(diff: np.ndarray, resamples: int, seed: int) -> list[float]:
    """95% interval of the mean of per-state differences, resampling states."""
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), (resamples, len(diff)))].mean(axis=1)
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


def main() -> None:
    args = parse_args()
    arms = dict(entry.split("=", 1) for entry in args.arms)
    if args.reference not in arms:
        raise ValueError(
            f"Expected --reference among {list(arms)}, got {args.reference}."
        )
    records = {name: load_records(Path(run)) for name, run in arms.items()}
    task_of = {r["state_id"]: r["task_id"] for r in next(iter(records.values()))}
    task_index = {r["task"]: r["task_id"] for r in next(iter(records.values()))}
    states = sorted(task_of)
    for name, rows in records.items():
        if sorted({r["state_id"] for r in rows}) != states:
            raise ValueError(f"Expected arm {name} on the same states as the others.")
    tasks = sorted(set(task_of.values()))
    reruns = max(r["rerun"] for r in records[args.reference]) + 1
    per_task_episodes = len(records[args.reference]) // len(tasks)
    title = f"Test SR ({reruns} reruns, {per_task_episodes} episodes per task)"

    report, lines = {"arms": arms, "reference": args.reference}, []
    for metric, label in METRICS.items():
        rates = {
            name: np.array(list(state_rates(rows, metric).values()))
            for name, rows in records.items()
        }
        task_ids = np.array([task_of[s] for s in states])
        task_sr = {
            name: {t: 100 * float(r[task_ids == t].mean()) for t in tasks}
            for name, r in rates.items()
        }
        overall = {
            name: float(np.mean(list(v.values()))) for name, v in task_sr.items()
        }
        deltas = {}
        names = list(arms)
        for name in names[names.index(args.reference) + 1 :]:
            r = rates[name]
            diff = 100 * (r - rates[args.reference])
            low, high = bootstrap(diff, args.resamples, args.seed)
            deltas[name] = {"mean": float(diff.mean()), "ci95": [low, high]}
        report[metric] = {
            "all": overall,
            "per_task": task_sr,
            "minus_reference": deltas,
        }

        lines += [f"## {label}", ""]
        lines += table(
            [title, *arms],
            [["**All 10**", *(f"**{overall[n]:.1f}%**" for n in arms)]],
        )
        lines.append("")
        for name, d in deltas.items():
            low, high = d["ci95"]
            lines.append(
                f"- {name} minus {args.reference}: {d['mean']:+.1f} [{low:+.1f}, {high:+.1f}]"
            )
        lines.append("")
        lines += table(
            ["Task", *(str(t) for t in tasks)],
            [[n, *(f"{task_sr[n][t]:.0f}" for t in tasks)] for n in arms],
        )
        lines.append("")

    if args.selections:
        kept = {}
        for path in args.selections:
            selection = json.loads((Path(path) / "selection.json").read_text())
            for task, v in selection["per_task"].items():
                kept[task_index[task]] = v
        report["kept"] = kept
        lines += ["## Kept imagined data", ""]
        lines += table(
            ["Task", "States", "Episodes", "Frames"],
            [
                [str(t), str(v["states"]), str(v["episodes"]), str(v["frames"])]
                for t, v in sorted(kept.items())
            ],
        )
        lines.append("")

    if args.fidelity:
        open_loop = json.loads(Path(args.fidelity).read_text())["open_loop"]
        report["open_loop"] = open_loop
        keys = (
            "sim_success_rate",
            "real_judge_success_rate",
            "replay_judge_success_rate",
            "disagreement",
            "fake_success",
            "missed_success",
        )
        rows = [
            [t, *(f"{100 * v[k]:.1f}%" for k in keys)]
            for t, v in open_loop["per_task"].items()
        ]
        rows.append(["All", *(f"{100 * open_loop[k]:.1f}%" for k in keys)])
        lines += ["## Open-loop disagreement", ""]
        lines += table(
            [
                "Task",
                "Real SR (sim)",
                "Real SR (judge)",
                "Replay SR (judge)",
                "Disagreement",
                "Fake successes",
                "Missed successes",
            ],
            rows,
        )
        lines.append("")

    out = Path(args.out)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    out.with_suffix(".md").write_text("\n".join(lines))
    logger.info("\n".join(lines))


if __name__ == "__main__":
    main()
