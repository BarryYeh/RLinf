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

"""Write the Robometer-judged successes of imagined rollouts as a LeRobot v2 dataset.

``--rollouts`` is an ``imagine_rollouts.py`` output that ``robometer_judge.py`` has
labeled. Of each state's successful rollouts, the ``--keep`` with the highest
``mean_score`` are kept, each cut at its ``success_step``: row ``t`` holds frame ``t``
and the action taken from it. The dataset at ``--out`` has the ``image`` and
``actions`` features ``LeRobotFrameDataset`` reads, PNG-encoded like
physical-intelligence/libero, and ``selection.json`` lists the kept episodes.
"""

import argparse
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import PIL.Image
import pyarrow as pa
import pyarrow.parquet as pq

from rlinf.utils.logging import get_logger

logger = get_logger()

CHUNKS_SIZE = 1000
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", required=True, help="Judged imagined rollouts.")
    parser.add_argument("--out", required=True, help="Output LeRobot dataset dir.")
    parser.add_argument("--keep", type=int, default=4, help="Episodes kept per state.")
    return parser.parse_args()


def png(frame: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    PIL.Image.fromarray(frame).save(buffer, format="PNG")
    return buffer.getvalue()


def main() -> None:
    args = parse_args()
    rollouts, out = Path(args.rollouts), Path(args.out)
    if out.exists():
        raise FileExistsError(f"Expected a new output directory, found {out}.")

    rows = [json.loads(line) for line in open(rollouts / "robometer_judge.jsonl")]
    by_state = defaultdict(list)
    for row in rows:
        if row["success"]:
            by_state[row["state_id"]].append(row)
    kept = []
    for state_id in sorted(by_state):
        ranked = sorted(by_state[state_id], key=lambda r: -r["mean_score"])
        kept.extend(ranked[: args.keep])
    states = {r["state_id"] for r in rows}
    logger.info(
        f"{sum(r['success'] for r in rows)}/{len(rows)} rollouts succeed; "
        f"{len(by_state)}/{len(states)} states have a success; keeping {len(kept)}"
    )

    tasks = sorted({r["task"] for r in kept})
    task_index = {task: i for i, task in enumerate(tasks)}
    episodes, index = [], 0
    for episode_index, row in enumerate(kept):
        with np.load(rollouts / "episodes" / f"{row['episode']}.npz") as episode:
            length = row["success_step"]
            frames = episode["frames"][:length]
            actions = episode["actions"][:length].astype(np.float32)
        table = pa.table(
            {
                "image": pa.array(
                    [
                        {"bytes": png(frame), "path": f"frame_{t:06d}.png"}
                        for t, frame in enumerate(frames)
                    ],
                    type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
                ),
                "actions": pa.FixedSizeListArray.from_arrays(
                    pa.array(actions.reshape(-1)), 7
                ),
                "frame_index": pa.array(np.arange(length), type=pa.int64()),
                "episode_index": pa.array(np.full(length, episode_index), pa.int64()),
                "index": pa.array(np.arange(index, index + length), type=pa.int64()),
                "task_index": pa.array(
                    np.full(length, task_index[row["task"]]), pa.int64()
                ),
            }
        )
        path = out / DATA_PATH.format(
            episode_chunk=episode_index // CHUNKS_SIZE, episode_index=episode_index
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        index += length
        episodes.append(
            {
                "episode_index": episode_index,
                "tasks": [row["task"]],
                "length": length,
                "source": row["episode"],
                "state_id": row["state_id"],
                "mean_score": row["mean_score"],
            }
        )

    meta = out / "meta"
    meta.mkdir(parents=True)
    info = {
        "codebase_version": "v2.0",
        "robot_type": "panda",
        "total_episodes": len(episodes),
        "total_frames": index,
        "total_tasks": len(tasks),
        "total_videos": 0,
        "total_chunks": (len(episodes) - 1) // CHUNKS_SIZE + 1,
        "chunks_size": CHUNKS_SIZE,
        "fps": 10,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": DATA_PATH,
        "features": {
            "image": {
                "dtype": "image",
                "shape": [256, 256, 3],
                "names": ["height", "width", "channel"],
            },
            "actions": {"dtype": "float32", "shape": [7], "names": ["actions"]},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (meta / "info.json").write_text(json.dumps(info, indent=2))
    with open(meta / "episodes.jsonl", "w") as f:
        f.writelines(json.dumps(ep) + "\n" for ep in episodes)
    with open(meta / "tasks.jsonl", "w") as f:
        f.writelines(
            json.dumps({"task_index": i, "task": task}) + "\n"
            for i, task in enumerate(tasks)
        )

    per_task = defaultdict(lambda: {"states": set(), "episodes": 0, "frames": 0})
    for ep in episodes:
        entry = per_task[ep["tasks"][0]]
        entry["states"].add(ep["state_id"])
        entry["episodes"] += 1
        entry["frames"] += ep["length"]
    selection = {
        "rollouts": str(rollouts.resolve()),
        "keep": args.keep,
        "episodes": len(episodes),
        "frames": index,
        "per_task": {
            task: {**v, "states": len(v["states"])} for task, v in per_task.items()
        },
        "kept": [ep["source"] for ep in episodes],
    }
    (out / "selection.json").write_text(json.dumps(selection, indent=2))
    logger.info(f"Wrote {len(episodes)} episodes, {index} frames to {out}")


if __name__ == "__main__":
    main()
