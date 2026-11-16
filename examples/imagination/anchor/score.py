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

"""Failure scores of real episodes, the methods ``select.py --method`` picks.

A score rates every chunk boundary of every episode of a ``collect_rollouts.py`` run
from what the policy saw there: its frame history and the instruction. Higher means
closer to failing. ``SCORES`` maps a method name to its class; a new method
subclasses ``AnchorScore`` and is added there.
"""

import argparse
import copy
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Iterator

import numpy as np
import torch
from torch import nn

from rlinf.utils.logging import get_logger

from ..common import episode_name, load_policy, load_records

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

    ``args`` is the ``select.py`` namespace: the shared ``ckpt``,
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


class RandomNetworkDistillation(nn.Module):
    """FIPER's RND-OE networks (``rnd/rnd_models.py:RND_OE``); the forward is the error.

    The predictor learns to match the frozen, randomly initialized target, so its
    error is small on inputs like its training data.
    """

    def __init__(self, input_dim: int, output_dim: int = 512):
        super().__init__()
        self.target = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, output_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
        )
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, np.sqrt(2))
                nn.init.zeros_(layer.bias)
        self.target.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.predictor(x) - self.target(x)).norm(dim=-1)


class RndObservationEmbedding(AnchorScore):
    """FIPER's RND-OE: random network distillation on the policy's observation
    embedding, here its denoiser condition (``encode_obs``: frame-history CLS
    features and instruction features).

    The predictor trains on every frame of the ``--rnd-task-suite`` demonstrations
    in ``--rnd-demos`` (with the frame history the policy trains on). The successes
    of ``--rnd-val-run``, a ``collect_rollouts.py`` run on other initial states, at
    their boundaries before success, pick the best epoch and stop training after
    ``--rnd-patience`` epochs without improvement. Optimizer and schedule follow
    FIPER's ``configs/eval/rnd_parameters.yaml``.
    """

    name = "rnd_oe"
    LR, LR_MIN, WEIGHT_DECAY, BATCH_SIZE = 1e-4, 1e-6, 1e-5, 256

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        missing = [
            f"--rnd-{k.replace('_', '-')}"
            for k in ("demos", "val_run", "cache_dir")
            if getattr(args, f"rnd_{k}") is None
        ]
        if missing:
            raise ValueError(f"Expected {', '.join(missing)} with --method rnd_oe.")

    @staticmethod
    def add_args(group: argparse._ArgumentGroup) -> None:
        group.add_argument("--rnd-demos", help="LeRobot v2 demonstrations (training).")
        group.add_argument(
            "--rnd-val-run", help="collect_rollouts.py run whose successes validate."
        )
        group.add_argument("--rnd-cache-dir", help="Decoded demo frame cache.")
        group.add_argument("--rnd-task-suite", default="libero_spatial")
        group.add_argument("--rnd-epochs", type=int, default=250)
        group.add_argument("--rnd-patience", type=int, default=7)

    def settings(self) -> dict:
        args = self.args
        return {
            "demos": os.path.abspath(args.rnd_demos),
            "val_run": os.path.abspath(args.rnd_val_run),
            "task_suite": args.rnd_task_suite,
            "epochs": args.rnd_epochs,
            "patience": args.rnd_patience,
        }

    @torch.no_grad()
    def _encode(self, policy, batches) -> torch.Tensor:
        return torch.cat(
            [policy.encode_obs(images, tasks) for images, tasks in batches]
        )

    def _demo_embeddings(self, policy) -> torch.Tensor:
        from rlinf.data.datasets.diffusion_policy.lerobot_dataset import (
            LeRobotFrameDataset,
            libero_suite_tasks,
        )

        cfg = policy.cfg
        dataset = LeRobotFrameDataset(
            root=self.args.rnd_demos,
            n_obs_steps=cfg.n_obs_steps,
            obs_stride=cfg.num_action_chunks,
            horizon=cfg.horizon,
            action_dim=cfg.action_dim,
            cache_dir=self.args.rnd_cache_dir,
            tasks=libero_suite_tasks(self.args.rnd_task_suite),
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.args.batch_size, num_workers=8
        )
        logger.info(f"rnd_oe: embedding {len(dataset)} demonstration frames")
        return self._encode(policy, ((b["images"], b["task"]) for b in loader))

    def _run_embeddings(self, policy, run: RealRun) -> torch.Tensor:
        """``[episodes, num_chunks, D]`` embeddings of every boundary of ``run``."""
        per_call = max(1, self.args.batch_size // run.num_chunks)
        batches = (
            (torch.from_numpy(histories), tasks)
            for _, histories, tasks in run.boundaries(policy.cfg.n_obs_steps, per_call)
        )
        return self._encode(policy, batches).view(len(run.records), run.num_chunks, -1)

    def score(self, run: RealRun) -> np.ndarray:
        args = self.args
        policy = load_policy(args.model_cfg, args.ckpt, run.num_chunks)
        train = self._demo_embeddings(policy).float()
        val_run = RealRun(args.rnd_val_run, run.chunk)
        val_all = self._run_embeddings(policy, val_run)
        val = torch.cat(
            [
                val_all[i, : val_run.pre_success_chunks(r)]
                for i, r in enumerate(val_run.records)
                if r["success"]
            ]
        ).float()
        embeddings = self._run_embeddings(policy, run).float()
        del policy

        model = self._fit(train, val)
        with torch.no_grad():
            errors = torch.cat([model(x) for x in embeddings.flatten(0, 1).split(4096)])
            errors = errors.view(len(run.records), run.num_chunks).cpu().numpy()
            gap = {
                "demos": self._mean_error(model, train),
                "validation successes": self._mean_error(model, val),
            }
        success = np.array([r["success"] for r in run.records])
        before = [
            errors[i, : run.pre_success_chunks(r)]
            for i, r in enumerate(run.records)
            if r["success"]
        ]
        gap["scored successes"] = float(np.concatenate(before).mean())
        gap["scored failures"] = float(errors[~success].mean())
        logger.info(
            "rnd_oe: mean error (successes before success) "
            + ", ".join(f"{k} {v:.4f}" for k, v in gap.items())
        )
        return errors

    @staticmethod
    def _mean_error(model: nn.Module, x: torch.Tensor) -> float:
        return float(torch.cat([model(b) for b in x.split(4096)]).mean())

    def _fit(self, train: torch.Tensor, val: torch.Tensor) -> nn.Module:
        """Train the predictor on ``train``; return the best weights on ``val``."""
        args = self.args
        torch.manual_seed(args.seed)
        model = RandomNetworkDistillation(train.shape[1]).to(train.device)
        optimizer = torch.optim.AdamW(
            model.predictor.parameters(),
            lr=self.LR,
            eps=1e-8,
            weight_decay=self.WEIGHT_DECAY,
        )
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.rnd_epochs, eta_min=self.LR_MIN
        )
        logger.info(
            f"rnd_oe: training on {len(train)} demo frames, validating on "
            f"{len(val)} success boundaries"
        )
        best, best_state, stale = float("inf"), None, 0
        for epoch in range(args.rnd_epochs):
            model.train()
            losses = []
            for ids in torch.randperm(len(train), device=train.device).split(
                self.BATCH_SIZE
            ):
                loss = model(train[ids]).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            schedule.step()
            model.eval()
            with torch.no_grad():
                val_loss = self._mean_error(model, val)
            if val_loss < best:
                best, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
            if epoch % 10 == 0 or stale >= args.rnd_patience:
                logger.info(
                    f"rnd_oe: epoch {epoch} train {np.mean(losses):.4f} "
                    f"validation {val_loss:.4f} (best {best:.4f})"
                )
            if stale >= args.rnd_patience:
                logger.info(
                    f"rnd_oe: stopped at epoch {epoch}, best validation {best:.4f}"
                )
                break
        model.load_state_dict(best_state)
        return model.eval()


SCORES: dict[str, type[AnchorScore]] = {
    "ace": ActionChunkEntropy,
    "rnd_oe": RndObservationEmbedding,
}
