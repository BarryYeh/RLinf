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

"""Compare the failure anchors of several ``select.py`` outputs.

For each ``anchors.json`` (labeled by its methods, or by ``--labels``), prints as
Markdown:

- separation: the AUROC of an episode's largest ratio of score to threshold (1 is at
  threshold for every method) within its first ``H`` steps, failures against
  successes that have not succeeded by step ``H`` (both still trying), for several
  ``H``;
- alarms: failures alarmed, successes alarmed before their first success, and the
  anchor steps of failures, overall and per task.
"""

import argparse
import json
from pathlib import Path

import numpy as np

HORIZONS = (64, 96, 128, 160, 192)


def auroc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Probability that a positive outranks a negative, ties counting half."""
    if not len(positive) or not len(negative):
        return float("nan")
    greater = (positive[:, None] > negative[None]).mean()
    ties = (positive[:, None] == negative[None]).mean()
    return float(greater + 0.5 * ties)


class Anchors:
    def __init__(self, path: Path, label: str | None):
        data = json.loads(path.read_text())
        combine = f" {data['settings']['combine']} "
        self.label = label or combine.join(data["methods"])
        rows = data["episodes"]
        self.rows = rows
        self.chunk = data["chunk"]
        self.ratio = np.array([r["ratio"] for r in rows])
        self.success = np.array([r["success"] for r in rows])
        self.first_success = np.array([r["first_success_step"] for r in rows])
        self.task = np.array([r["task_id"] for r in rows])

    def separation(self, steps: int) -> str:
        chunks = steps // self.chunk
        peak = self.ratio[:, :chunks].max(axis=1)
        trying = self.success & (self.first_success > steps)
        value = auroc(peak[~self.success], peak[trying])
        return f"{value:.3f} ({trying.sum()})"

    def alarms(self, mask: np.ndarray) -> str:
        rows = [r for r, keep in zip(self.rows, mask) if keep]
        failed = [r for r in rows if not r["success"]]
        succeeded = [r for r in rows if r["success"]]
        alarmed = sum(r["anchor_reason"] == "alarm" for r in failed)
        early = sum(0 <= r["alarm_step"] < r["first_success_step"] for r in succeeded)
        anchors = [r["anchor_step"] for r in failed]
        where = (
            f"{np.median(anchors):.0f} ({np.percentile(anchors, 25):.0f}-"
            f"{np.percentile(anchors, 75):.0f})"
            if anchors
            else "-"
        )
        return f"{alarmed}/{len(failed)} | {early}/{len(succeeded)} | {where}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("anchors", nargs="+", type=Path, help="anchors.json files.")
    parser.add_argument("--labels", nargs="+", help="One label per file.")
    args = parser.parse_args()
    if args.labels and len(args.labels) != len(args.anchors):
        raise ValueError(
            f"Expected one label per file ({len(args.anchors)}), got "
            f"{len(args.labels)}."
        )
    labels = args.labels or [None] * len(args.anchors)
    runs = [Anchors(path, label) for path, label in zip(args.anchors, labels)]

    print("## Separation (AUROC; failures vs. successes still trying at step H)\n")
    print("In parentheses: successes still trying.\n")
    print("| Method | " + " | ".join(f"H = {h}" for h in HORIZONS) + " |")
    print("|---" * (len(HORIZONS) + 1) + "|")
    for run in runs:
        values = " | ".join(run.separation(h) for h in HORIZONS)
        print(f"| {run.label} | {values} |")

    header = "Failures alarmed | Successes alarmed early | Anchor step, median (IQR)"
    print("\n## Alarms, all tasks\n")
    print(f"| Method | {header} |\n|---|---|---|---|")
    for run in runs:
        print(f"| {run.label} | {run.alarms(np.ones(len(run.rows), bool))} |")

    print("\n## Alarms per task\n")
    print(f"| Task | Method | {header} |\n|---|---|---|---|---|")
    for task in sorted(set(runs[0].task.tolist())):
        for run in runs:
            print(f"| {task} | {run.label} | {run.alarms(run.task == task)} |")


if __name__ == "__main__":
    main()
