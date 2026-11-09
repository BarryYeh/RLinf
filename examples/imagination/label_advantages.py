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

"""Label action chunks by an n-step advantage built from Robometer success scores.

Each ``--datasets`` entry is an ``export_imagined.py --all`` dataset, whose
``meta/episodes.jsonl`` holds every episode's Robometer ``success_prob`` at
``score_steps`` (one score per chunk boundary) and its success window. Chunk ``c``
covers frames ``c * chunk`` to ``(c + 1) * chunk - 1`` and is scored by the return
``G_c = p[c + 1] + ... + p[c + lookahead]``, where ``p`` is 1 from the success window
on (success is absorbing) and repeats its last score past the episode end. The
advantage is ``G_c`` minus the mean ``G`` of the task's chunks, and the top
``--positive-quantile`` of each task's chunks are positive.

Writes ``meta/advantages_<tag>.parquet`` (``episode_index``, ``frame_index``,
``advantage_continuous``, ``advantage``; one row per frame) into each dataset, as
``LeRobotFrameDataset`` reads it, and ``meta/advantages_<tag>.json`` with the settings
and per-task statistics.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from rlinf.utils.logging import get_logger

logger = get_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--tag", required=True, help="Writes advantages_<tag>.*.")
    parser.add_argument(
        "--lookahead", type=int, default=4, help="Chunks summed in the return."
    )
    parser.add_argument("--positive-quantile", type=float, default=0.3)
    parser.add_argument("--chunk", type=int, default=8, help="Frames per chunk.")
    return parser.parse_args()


def chunk_returns(episode: dict, chunk: int, lookahead: int) -> np.ndarray:
    """``G_c`` for every chunk of one truncated episode."""
    steps = episode["score_steps"]
    if steps[1] - steps[0] != chunk:
        raise ValueError(
            f"Expected Robometer scores every {chunk} steps, got stride "
            f"{steps[1] - steps[0]} in episode {episode['episode_index']}."
        )
    if episode["length"] % chunk != 0:
        raise ValueError(
            f"Expected an episode length divisible by {chunk}, got "
            f"{episode['length']} in episode {episode['episode_index']}."
        )
    p = np.asarray(episode["success_prob"], dtype=np.float64)
    if episode["success"]:
        p[episode["success_step"] // chunk :] = 1.0
    p = np.concatenate([p, np.full(lookahead, p[-1])])
    num_chunks = episode["length"] // chunk
    return np.array([p[c + 1 : c + 1 + lookahead].sum() for c in range(num_chunks)])


def main() -> None:
    args = parse_args()
    roots = [Path(d) for d in args.datasets]
    episodes = {
        root: [json.loads(line) for line in open(root / "meta/episodes.jsonl")]
        for root in roots
    }
    returns = {
        (root, ep["episode_index"]): chunk_returns(ep, args.chunk, args.lookahead)
        for root in roots
        for ep in episodes[root]
    }

    by_task = defaultdict(list)
    for root in roots:
        for ep in episodes[root]:
            by_task[ep["tasks"][0]].append(returns[root, ep["episode_index"]])
    stats = {}
    for task, task_returns in by_task.items():
        G = np.concatenate(task_returns)
        baseline = float(G.mean())
        threshold = float(np.quantile(G - baseline, 1.0 - args.positive_quantile))
        stats[task] = {
            "baseline": baseline,
            "threshold": threshold,
            "chunks": int(len(G)),
            "positive_chunks": int((G - baseline >= threshold).sum()),
        }

    for root in roots:
        columns = defaultdict(list)
        for ep in episodes[root]:
            task = stats[ep["tasks"][0]]
            advantage = returns[root, ep["episode_index"]] - task["baseline"]
            per_frame = np.repeat(advantage, args.chunk)
            columns["episode_index"].append(
                np.full(len(per_frame), ep["episode_index"])
            )
            columns["frame_index"].append(np.arange(len(per_frame)))
            columns["advantage_continuous"].append(per_frame)
            columns["advantage"].append(per_frame >= task["threshold"])
        table = pa.table({k: np.concatenate(v) for k, v in columns.items()})
        pq.write_table(table, root / "meta" / f"advantages_{args.tag}.parquet")
        settings = {
            "lookahead": args.lookahead,
            "positive_quantile": args.positive_quantile,
            "chunk": args.chunk,
            "datasets": [str(r.resolve()) for r in roots],
            "per_task": stats,
        }
        (root / "meta" / f"advantages_{args.tag}.json").write_text(
            json.dumps(settings, indent=2)
        )
        logger.info(
            f"{root}: {int(table['advantage'].to_numpy().sum())}/{table.num_rows} "
            f"frames positive"
        )
    for task, s in sorted(stats.items()):
        logger.info(
            f"{task}: {s['positive_chunks']}/{s['chunks']} chunks positive, "
            f"baseline {s['baseline']:.3f}, threshold {s['threshold']:.3f}"
        )


if __name__ == "__main__":
    main()
