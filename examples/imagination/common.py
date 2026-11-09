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

"""Rollout files shared by the imagination scripts.

A rollout run is a directory with ``episodes/<episode_name>.npz`` (one per episode,
holding its arrays and its record fields) and ``summary.json`` (success rates and
``episodes_list``, the records).
"""

import json
from pathlib import Path

import numpy as np


def parse_ids(spec: str) -> list[int]:
    """Ids from ``'a-b'`` (inclusive) or a comma list."""
    if "-" in spec:
        start, end = (int(x) for x in spec.split("-"))
        return list(range(start, end + 1))
    return [int(x) for x in spec.split(",")]


def episode_name(state_id: int, rerun: int) -> str:
    return f"state_{state_id:03d}_rerun_{rerun}"


def load_policy(model_cfg: str, ckpt: str, num_chunks_per_episode: int):
    """The policy of ``model_cfg`` with SFT weights ``ckpt``, on CUDA in eval mode."""
    import torch
    from omegaconf import OmegaConf

    from rlinf.models import get_model

    cfg = OmegaConf.load(model_cfg)
    cfg.obs_history_reset_interval = num_chunks_per_episode
    policy = get_model(cfg)
    policy.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return policy.to("cuda").eval()


def save_episode(out: Path, record: dict, **arrays: np.ndarray) -> None:
    np.savez_compressed(
        out / "episodes" / f"{episode_name(record['state_id'], record['rerun'])}.npz",
        **arrays,
        **{k: np.asarray(v) for k, v in record.items()},
    )


def write_summary(out: Path, records: list[dict], **meta) -> dict:
    """Write ``summary.json``: ``meta``, success rates overall and per task, records."""
    per_task = {}
    for task_id in sorted({r["task_id"] for r in records}):
        rows = [r for r in records if r["task_id"] == task_id]
        per_task[task_id] = {
            "task": rows[0]["task"],
            "episodes": len(rows),
            "success_rate": float(np.mean([r["success"] for r in rows])),
        }
    summary = {
        **meta,
        "episodes": len(records),
        "success_rate": float(np.mean([r["success"] for r in records])),
        "success_at_end_rate": float(np.mean([r["success_at_end"] for r in records])),
        "per_task": per_task,
        "episodes_list": records,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def load_records(run: Path) -> list[dict]:
    path = Path(run) / "summary.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Expected a finished rollout run with {path}, found none."
        )
    return json.loads(path.read_text())["episodes_list"]
