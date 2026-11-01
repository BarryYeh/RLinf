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

import math
from typing import Optional

import torch
import torch.nn as nn


def squaredcos_cap_v2_betas(num_train_timesteps: int) -> torch.Tensor:
    """Cosine beta schedule (Nichol & Dhariwal), as diffusers' ``squaredcos_cap_v2``."""

    def alpha_bar(t: float) -> float:
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = [
        min(
            1
            - alpha_bar((i + 1) / num_train_timesteps)
            / alpha_bar(i / num_train_timesteps),
            0.999,
        )
        for i in range(num_train_timesteps)
    ]
    return torch.tensor(betas, dtype=torch.float32)


class DDPMScheduler(nn.Module):
    """Epsilon-prediction DDPM with the ``fixed_small`` variance and sample clipping.

    Matches diffusers' ``DDPMScheduler`` as configured by LeRobot's diffusion
    policy, including strided sampling when fewer inference steps are used.
    """

    def __init__(self, num_train_timesteps: int, clip_sample_range: float = 1.0):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.clip_sample_range = clip_sample_range
        alphas_cumprod = torch.cumprod(
            1.0 - squaredcos_cap_v2_betas(num_train_timesteps), dim=0
        )
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)

    def add_noise(
        self, x0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        acp = self.alphas_cumprod[timesteps].view(-1, *([1] * (x0.ndim - 1)))
        return acp.sqrt() * x0 + (1 - acp).sqrt() * noise

    def timesteps(self, num_inference_steps: int) -> list[int]:
        if not 0 < num_inference_steps <= self.num_train_timesteps:
            raise ValueError(
                f"Expected 0 < num_inference_steps <= {self.num_train_timesteps}, "
                f"got {num_inference_steps}."
            )
        step_ratio = self.num_train_timesteps // num_inference_steps
        return [i * step_ratio for i in reversed(range(num_inference_steps))]

    def step(
        self,
        eps: torch.Tensor,
        t: int,
        sample: torch.Tensor,
        num_inference_steps: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        prev_t = t - self.num_train_timesteps // num_inference_steps
        acp_t = self.alphas_cumprod[t]
        acp_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else acp_t.new_tensor(1.0)
        beta_prod_t = 1 - acp_t
        alpha_t = acp_t / acp_prev
        beta_t = 1 - alpha_t

        x0 = (sample - beta_prod_t.sqrt() * eps) / acp_t.sqrt()
        x0 = x0.clamp(-self.clip_sample_range, self.clip_sample_range)
        mean = (acp_prev.sqrt() * beta_t / beta_prod_t) * x0 + (
            alpha_t.sqrt() * (1 - acp_prev) / beta_prod_t
        ) * sample
        if t == 0:
            return mean
        variance = ((1 - acp_prev) / beta_prod_t * beta_t).clamp(min=1e-20)
        noise = torch.randn(
            sample.shape, generator=generator, device=sample.device, dtype=sample.dtype
        )
        return mean + variance.sqrt() * noise
