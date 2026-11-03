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

from typing import Optional

import torch


class FrameHistory:
    """Per-slot history of the last ``n_obs_steps`` frames across rollout calls.

    The env worker returns only the last frame of each action chunk, so a policy
    that conditions on several frames keeps them itself, keyed by batch slot. A
    slot must map to the same env across calls, and a new episode must start every
    ``reset_interval`` calls (``None`` never resets): a fresh history repeats the
    current frame.
    """

    def __init__(self, n_obs_steps: int, reset_interval: Optional[int] = None):
        self.n_obs_steps = n_obs_steps
        self.reset_interval = reset_interval
        self.reset()

    def reset(self) -> None:
        """Forget the frames of every slot."""
        self._frames: Optional[torch.Tensor] = None
        self._num_calls = 0

    def update(self, frames: torch.Tensor) -> torch.Tensor:
        """Append ``[B, H, W, 3]`` frames and return the ``[B, S, H, W, 3]`` history."""
        if (
            self._frames is None
            or self._frames.shape[0] != frames.shape[0]
            or (
                self.reset_interval is not None
                and self._num_calls % self.reset_interval == 0
            )
        ):
            self._frames = frames.unsqueeze(1).repeat(1, self.n_obs_steps, 1, 1, 1)
        else:
            self._frames = torch.cat([self._frames[:, 1:], frames.unsqueeze(1)], dim=1)
        self._num_calls += 1
        return self._frames
