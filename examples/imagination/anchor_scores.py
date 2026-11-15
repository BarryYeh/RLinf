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

"""Failure scores of real episodes, the methods ``select_anchors.py --method`` picks.

A score rates every chunk boundary of every episode of a ``collect_rollouts.py`` run
from what the policy saw there: its frame history and the instruction. Higher means
closer to failing. ``SCORES`` maps a method name to its class; a new method
subclasses ``AnchorScore`` and is added there.
"""

import argparse
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Iterator

import numpy as np
import torch
from common import episode_name, load_policy, load_records

from rlinf.utils.logging import get_logger

logger = get_logger()


class RealRun:
    """A finished ``collect_rollouts.py`` run, read at its chunk boundaries.

    Boundary ``c`` is the policy call before step ``c * chunk``; every episode has
    ``num_chunks`` of them.
    """

    def __init__(self, path: str, chunk: int):
        self.path = Path(path)
        self.records = load_records(self.path)
        self.chunk = chunk
        with np.load(self.episode_path(self.records[0])) as episode:
            num_steps = len(episode["actions"])
        if num_steps % chunk:
            raise ValueError(
                f"Expected episodes of {self.path} a multiple of {chunk} steps, got "
                f"{num_steps}."
            )
        self.num_chunks = num_steps // chunk

    @property
    def names(self) -> list[str]:
        return [episode_name(r["state_id"], r["rerun"]) for r in self.records]

    def episode_path(self, record: dict) -> Path:
        name = episode_name(record["state_id"], record["rerun"])
        return self.path / "episodes" / f"{name}.npz"

    def pre_success_chunks(self, record: dict) -> int:
        """How many leading chunks a successful episode acts before its first success."""
        return min(-(-record["first_success_step"] // self.chunk), self.num_chunks)

    def boundaries(
        self, n_obs_steps: int, episodes_per_call: int
    ) -> Iterator[tuple[int, np.ndarray, list[str]]]:
        """Yield ``(first episode, histories, instructions)`` for consecutive episodes.

        ``histories`` is ``[episodes * num_chunks, n_obs_steps, H, W, 3]``,
        episode-major; a fresh history repeats the reset frame, as ``FrameHistory``
        does.
        """
        calls = np.arange(self.num_chunks)[:, None] + np.arange(1 - n_obs_steps, 1)
        frame_ids = np.clip(calls, 0, None) * self.chunk
        for start in range(0, len(self.records), episodes_per_call):
            histories, instructions = [], []
            for record in self.records[start : start + episodes_per_call]:
                with np.load(self.episode_path(record)) as episode:
                    if len(episode["frames"]) != self.num_chunks * self.chunk + 1:
                        raise ValueError(
                            f"Expected {self.num_chunks * self.chunk + 1} frames in "
                            f"{self.episode_path(record)}, got {len(episode['frames'])}."
                        )
                    histories.append(episode["frames"][frame_ids])
                    instructions += [str(episode["task"])] * self.num_chunks
            yield start, np.concatenate(histories), instructions


class AnchorScore(ABC):
    """A failure score. Subclasses set ``name`` and implement the methods below.

    ``args`` is the ``select_anchors.py`` namespace: the shared ``ckpt``,
    ``model_cfg``, ``batch_size`` and ``seed``, plus the score's own options.
    """

    name: ClassVar[str]

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @staticmethod
    @abstractmethod
    def add_args(group: argparse._ArgumentGroup) -> None:
        """Add the score's options, each prefixed with ``--<name>-``."""

    @abstractmethod
    def settings(self) -> dict:
        """The score's options; with the checkpoint and seed, they key the cache."""

    @abstractmethod
    def score(self, run: RealRun) -> np.ndarray:
        """``[episodes, num_chunks]`` scores in ``run.records`` order."""


def histogram_entropy(points: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """FIPER's ACE estimator for ``points`` ``[..., samples, dims]``.

    The samples fall into a grid of ``cell``-sized cells that starts 1% of their
    spread below their minimum; returns the Shannon entropy (bits) of the cell
    counts, ``[...]``.
    """
    low, high = points.min(axis=-2, keepdims=True), points.max(axis=-2, keepdims=True)
    cells = np.floor((points - (low - 0.01 * (high - low))) / cell).astype(np.int64)
    base = int(cells.max()) + 1
    keys = np.zeros(cells.shape[:-1], dtype=np.int64)
    for d in range(cells.shape[-1]):
        keys = keys * base + cells[..., d]
    # Each sample's share of its cell; averaging -log2(share) over samples is the
    # entropy of the cell distribution.
    shares = (keys[..., :, None] == keys[..., None, :]).mean(axis=-1)
    return -np.log2(shares).mean(axis=-1)


class ActionChunkEntropy(AnchorScore):
    """FIPER's action-chunk entropy (ACE), ``third_party/fiper`` reference
    ``evaluation/method_eval_classes/entropy_eval.py``.

    At each boundary the policy draws ``--ace-samples`` chunks over its whole
    horizon. The cumulative sum of their xyz deltas is the end-effector path from
    the boundary; FIPER bins absolute positions, and the entropy does not depend on
    the offset. At each horizon step the samples fall into cells of
    ``--ace-cell-size`` times each dimension's range over the successful episodes'
    pre-success boundaries (a zero range takes the largest), and the score is the
    mean over steps of the entropy of the cell counts.
    """

    name = "ace"

    @staticmethod
    def add_args(group: argparse._ArgumentGroup) -> None:
        group.add_argument(
            "--ace-samples", type=int, default=32, help="Chunks drawn per boundary."
        )
        group.add_argument(
            "--ace-cell-size",
            type=float,
            default=0.03,
            help="Histogram cell size as a fraction of each dimension's range.",
        )

    def settings(self) -> dict:
        return {"samples": self.args.ace_samples, "cell_size": self.args.ace_cell_size}

    def score(self, run: RealRun) -> np.ndarray:
        paths = self._sample_paths(run)  # [E, C, H, S, 3]
        calibration = np.concatenate(
            [
                paths[i, : run.pre_success_chunks(r)].reshape(-1, 3)
                for i, r in enumerate(run.records)
                if r["success"]
            ]
        )
        spread = calibration.max(axis=0) - calibration.min(axis=0)
        cell = np.where(spread > 0, spread, spread.max()) * self.args.ace_cell_size
        logger.info(f"ace: cell size {np.round(cell, 4).tolist()}")
        return np.stack(
            [histogram_entropy(episode, cell).mean(axis=-1) for episode in paths]
        )

    def _sample_paths(self, run: RealRun) -> np.ndarray:
        """``[episodes, num_chunks, horizon, samples, 3]`` xyz paths from each boundary."""
        args, samples = self.args, self.args.ace_samples
        policy = load_policy(args.model_cfg, args.ckpt, run.num_chunks)
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        per_call = max(1, args.batch_size // (run.num_chunks * samples))
        paths = np.zeros(
            (len(run.records), run.num_chunks, policy.cfg.horizon, samples, 3),
            dtype=np.float32,
        )
        for start, histories, tasks in run.boundaries(policy.cfg.n_obs_steps, per_call):
            actions = policy.sample_actions(
                torch.from_numpy(histories), tasks, generator, num_samples=samples
            )
            xyz = actions[..., :3].cumsum(dim=1).float().cpu().numpy()
            xyz = xyz.reshape(-1, run.num_chunks, samples, *xyz.shape[1:])
            paths[start : start + len(xyz)] = xyz.transpose(0, 1, 3, 2, 4)
            if (start // per_call) % 50 == 0:
                logger.info(f"ace: sampled {start + len(xyz)}/{len(run.records)}")
        return paths


SCORES: dict[str, type[AnchorScore]] = {"ace": ActionChunkEntropy}
