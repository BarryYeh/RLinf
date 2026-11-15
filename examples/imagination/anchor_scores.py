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
from common import episode_name, load_records


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


SCORES: dict[str, type[AnchorScore]] = {}
