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

from typing import Any

from torch.utils.data import ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.data.datasets.diffusion_policy.lerobot_dataset import (
    LeRobotFrameDataset,
    build_diffusion_policy_datasets,
    libero_suite_tasks,
)
from rlinf.data.datasets.diffusion_policy.sampler import WeightedDatasetSampler


def build_diffusion_policy_sft_dataloader(
    cfg: Any, world_size: int, global_rank: int, data_paths: Any
) -> tuple[StatefulDataLoader, dict[str, int]]:
    """Build the SFT dataloader over every dataset in ``data_paths``.

    Without ``data.sampling_weights`` every frame is equally likely; with one weight
    per path, each path fills that share of the samples.
    """
    datasets = build_diffusion_policy_datasets(data_paths, cfg.actor.model, cfg.data)
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    weights = cfg.data.get("sampling_weights", None)
    if weights is None:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=True,
            seed=cfg.actor.seed,
        )
    else:
        sampler = WeightedDatasetSampler(
            [len(d) for d in datasets],
            list(weights),
            num_replicas=world_size,
            rank=global_rank,
            seed=cfg.actor.seed,
        )
    num_workers = cfg.data.get("num_workers", 8)
    data_loader = StatefulDataLoader(
        dataset,
        batch_size=cfg.actor.micro_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    return data_loader, {"num_samples": len(dataset)}


__all__ = [
    "LeRobotFrameDataset",
    "build_diffusion_policy_datasets",
    "build_diffusion_policy_sft_dataloader",
    "libero_suite_tasks",
]
