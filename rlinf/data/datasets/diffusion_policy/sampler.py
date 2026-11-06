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

from collections.abc import Iterator, Sequence

import torch
from torch.utils.data import Sampler


class WeightedDatasetSampler(Sampler[int]):
    """Draw indices of a ``ConcatDataset`` so each part fills a fixed share of samples.

    Each draw picks part ``i`` with probability ``weights[i] / sum(weights)`` and a
    uniform sample within it, with replacement. An epoch has as many draws as the
    concatenated dataset has samples, split across ranks; ``set_epoch`` reseeds.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        weights: Sequence[float],
        num_replicas: int,
        rank: int,
        seed: int,
    ):
        if len(lengths) != len(weights):
            raise ValueError(
                f"Expected one sampling weight per dataset ({len(lengths)}), got "
                f"{len(weights)}."
            )
        if any(w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError(
                f"Expected non-negative sampling weights with a positive sum, got "
                f"{list(weights)}."
            )
        self.sample_weights = torch.cat(
            [
                torch.full((length,), weight / length, dtype=torch.float64)
                for length, weight in zip(lengths, weights)
            ]
        )
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.num_samples = len(self.sample_weights) // num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.sample_weights,
            self.num_samples * self.num_replicas,
            replacement=True,
            generator=generator,
        )
        return iter(indices[self.rank :: self.num_replicas].tolist())
