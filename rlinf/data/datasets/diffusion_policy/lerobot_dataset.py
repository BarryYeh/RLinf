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

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from rlinf.utils.logging import get_logger

logger = get_logger()


def _normalize_task(task: str) -> str:
    return " ".join(task.lower().split())


def libero_suite_tasks(task_suite_name: str) -> set[str]:
    """Normalized language instructions of one LIBERO task suite."""
    from libero.libero import benchmark

    suites = benchmark.get_benchmark_dict()
    if task_suite_name not in suites:
        raise ValueError(
            f"Expected a LIBERO task suite in {sorted(suites)}, got {task_suite_name!r}."
        )
    suite = suites[task_suite_name]()
    return {_normalize_task(suite.get_task(i).language) for i in range(suite.n_tasks)}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _decode_image(value: Any) -> np.ndarray:
    import PIL.Image

    raw = value["bytes"] if isinstance(value, dict) else value
    with PIL.Image.open(io.BytesIO(raw)) as image:
        return np.asarray(image.convert("RGB"))


class LeRobotFrameDataset(Dataset):
    """Strided frame histories and action horizons from a LeRobot v2.x dataset.

    Sample ``t`` of an episode holds the frames at ``t - k * obs_stride`` for
    ``k = n_obs_steps - 1, ..., 0`` and the actions ``t, ..., t + horizon - 1``;
    indices outside the episode repeat its first frame or last action. Images
    must be stored in the parquet files (feature dtype ``image``). The selected
    frames are decoded once into a uint8 memmap under ``cache_dir`` so DataLoader
    workers share them through the page cache.
    """

    def __init__(
        self,
        root: str,
        n_obs_steps: int,
        obs_stride: int,
        horizon: int,
        action_dim: int,
        cache_dir: str,
        image_key: str = "image",
        action_key: str = "actions",
        tasks: Optional[set[str]] = None,
    ):
        self.root = Path(root).expanduser().resolve()
        self.n_obs_steps = n_obs_steps
        self.obs_stride = obs_stride
        self.horizon = horizon

        info = json.loads((self.root / "meta" / "info.json").read_text())
        version = str(info.get("codebase_version", ""))
        if not version.startswith("v2"):
            raise ValueError(
                f"Expected a LeRobot v2.x dataset at {self.root}, got codebase_version "
                f"{version!r}."
            )
        features = info["features"]
        for key in (image_key, action_key):
            if key not in features:
                raise KeyError(
                    f"Expected feature {key!r} in {self.root}, found {sorted(features)}."
                )
        if features[image_key]["dtype"] != "image":
            raise ValueError(
                f"Expected feature {image_key!r} with dtype 'image', got "
                f"{features[image_key]['dtype']!r}; video features are not supported."
            )
        if tuple(features[action_key]["shape"]) != (action_dim,):
            raise ValueError(
                f"Expected {action_key!r} of shape ({action_dim},), got "
                f"{tuple(features[action_key]['shape'])}."
            )

        episodes = _read_jsonl(self.root / "meta" / "episodes.jsonl")
        if tasks is not None:
            episodes = [
                ep
                for ep in episodes
                if any(_normalize_task(task) in tasks for task in ep["tasks"])
            ]
        if not episodes:
            raise ValueError(
                f"Expected episodes in {self.root} matching tasks "
                f"{sorted(tasks) if tasks is not None else 'any'}, found none."
            )
        self.parquet_paths = [self._parquet_path(info, ep) for ep in episodes]
        self.lengths = np.asarray([int(ep["length"]) for ep in episodes])
        self.starts = np.concatenate([[0], np.cumsum(self.lengths)[:-1]])
        self.ends = self.starts + self.lengths

        self.actions = self._load_actions(action_key)
        self.frames_path = self._build_frame_cache(
            image_key, tuple(features[image_key]["shape"]), Path(cache_dir)
        )
        self._frames: Optional[np.ndarray] = None
        logger.info(
            f"LeRobotFrameDataset: {len(episodes)} episodes, {len(self)} frames "
            f"from {self.root}"
        )

    def _parquet_path(self, info: dict[str, Any], episode: dict[str, Any]) -> Path:
        index = int(episode["episode_index"])
        return self.root / info["data_path"].format(
            episode_chunk=index // int(info["chunks_size"]), episode_index=index
        )

    def _load_actions(self, action_key: str) -> np.ndarray:
        import pyarrow.parquet as pq

        actions = []
        for path, length in zip(self.parquet_paths, self.lengths):
            column = pq.read_table(path, columns=[action_key]).column(action_key)
            episode_actions = np.asarray(column.to_pylist(), dtype=np.float32)
            if len(episode_actions) != length:
                raise ValueError(
                    f"Expected {length} frames in {path} as meta/episodes.jsonl "
                    f"states, found {len(episode_actions)}."
                )
            actions.append(episode_actions)
        return np.concatenate(actions)

    def _build_frame_cache(
        self, image_key: str, image_shape: tuple[int, ...], cache_dir: Path
    ) -> Path:
        key = json.dumps(
            [str(self.root), image_key, [str(p) for p in self.parquet_paths]]
        )
        entry = cache_dir / hashlib.sha1(key.encode()).hexdigest()[:16]
        frames_path = entry / "frames.npy"
        entry.mkdir(parents=True, exist_ok=True)
        with open(entry / ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not frames_path.exists():
                self._write_frames(image_key, image_shape, frames_path)
        return frames_path

    def _write_frames(
        self, image_key: str, image_shape: tuple[int, ...], frames_path: Path
    ) -> None:
        import pyarrow.parquet as pq

        logger.info(f"Decoding {int(self.lengths.sum())} frames into {frames_path}")
        tmp_path = frames_path.with_suffix(".tmp.npy")
        frames = np.lib.format.open_memmap(
            tmp_path,
            mode="w+",
            dtype=np.uint8,
            shape=(int(self.lengths.sum()), *image_shape),
        )

        def write_episode(i: int) -> None:
            column = pq.read_table(self.parquet_paths[i], columns=[image_key]).column(
                image_key
            )
            for t, value in enumerate(column.to_pylist()):
                image = _decode_image(value)
                if image.shape != image_shape:
                    raise ValueError(
                        f"Expected {image_key!r} frames of shape {image_shape} in "
                        f"{self.parquet_paths[i]}, got {image.shape} at row {t}."
                    )
                frames[self.starts[i] + t] = image

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
            list(pool.map(write_episode, range(len(self.parquet_paths))))
        frames.flush()
        os.replace(tmp_path, frames_path)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_frames"] = None
        return state

    def __len__(self) -> int:
        return int(self.ends[-1])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self._frames is None:
            self._frames = np.load(self.frames_path, mmap_mode="r")
        episode = int(np.searchsorted(self.ends, index, side="right"))
        start, length = int(self.starts[episode]), int(self.lengths[episode])
        t = index - start
        obs_t = [
            max(t - k * self.obs_stride, 0) for k in reversed(range(self.n_obs_steps))
        ]
        action_t = [min(t + k, length - 1) for k in range(self.horizon)]
        return {
            "images": torch.from_numpy(self._frames[[start + i for i in obs_t]]),
            "actions": torch.from_numpy(self.actions[[start + i for i in action_t]]),
        }


def build_diffusion_policy_datasets(
    data_paths: str | Sequence[str], model_cfg: Any, data_cfg: Any
) -> list[LeRobotFrameDataset]:
    """One dataset per path, sampled with the model's history and horizon."""
    paths = [data_paths] if isinstance(data_paths, str) else list(data_paths)
    task_suite_name = data_cfg.get("task_suite_name", None)
    tasks = libero_suite_tasks(task_suite_name) if task_suite_name else None
    return [
        LeRobotFrameDataset(
            root=path,
            n_obs_steps=model_cfg.n_obs_steps,
            obs_stride=model_cfg.num_action_chunks,
            horizon=model_cfg.horizon,
            action_dim=model_cfg.action_dim,
            cache_dir=data_cfg.cache_dir,
            image_key=data_cfg.get("image_key", "image"),
            action_key=data_cfg.get("action_key", "actions"),
            tasks=tasks,
        )
        for path in paths
    ]
