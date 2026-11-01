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

"""Networks of the LeRobot diffusion policy: a ResNet-18 keypoint encoder and a
FiLM-conditioned 1D UNet."""

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class SpatialSoftmax(nn.Module):
    """Reduce a feature map to the expected image coordinates of learned keypoints."""

    def __init__(self, channels: int, height: int, width: int, num_keypoints: int):
        super().__init__()
        self.keypoints = nn.Conv2d(channels, num_keypoints, kernel_size=1)
        self.num_keypoints = num_keypoints
        pos_y, pos_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height),
            torch.linspace(-1.0, 1.0, width),
            indexing="ij",
        )
        self.register_buffer(
            "pos_grid",
            torch.stack([pos_x.reshape(-1), pos_y.reshape(-1)], dim=1),
            persistent=False,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.keypoints(features).flatten(start_dim=2)
        attention = F.softmax(logits, dim=-1)
        return (attention @ self.pos_grid).flatten(start_dim=1)


class RgbEncoder(nn.Module):
    """ResNet-18 with GroupNorm, trained from scratch, pooled by spatial softmax.

    Takes images already resized and cropped to ``input_shape`` and normalized.
    """

    def __init__(self, input_shape: Sequence[int], num_keypoints: int):
        super().__init__()
        resnet = torchvision.models.resnet18(
            weights=None,
            norm_layer=lambda channels: nn.GroupNorm(channels // 16, channels),
        )
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        with torch.no_grad():
            _, channels, height, width = self.backbone(
                torch.zeros(1, 3, *input_shape)
            ).shape
        self.pool = SpatialSoftmax(channels, height, width, num_keypoints)
        self.feature_dim = num_keypoints * 2
        self.out = nn.Linear(self.feature_dim, self.feature_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return F.relu(self.out(self.pool(self.backbone(images))))


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        freqs = torch.exp(
            torch.arange(half_dim, device=x.device)
            * (-math.log(10000) / (half_dim - 1))
        )
        emb = x.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Conv1dBlock(nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, n_groups: int
    ):
        super().__init__(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )


class ConditionalResidualBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int,
        n_groups: int,
        use_film_scale_modulation: bool,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.use_film_scale_modulation = use_film_scale_modulation
        self.conv1 = Conv1dBlock(in_channels, out_channels, kernel_size, n_groups)
        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))
        self.conv2 = Conv1dBlock(out_channels, out_channels, kernel_size, n_groups)
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film_scale_modulation:
            scale, bias = cond_embed.split(self.out_channels, dim=1)
            out = scale * out + bias
        else:
            out = out + cond_embed
        return self.conv2(out) + self.residual_conv(x)


class ConditionalUnet1d(nn.Module):
    """Denoiser over an action sequence ``[B, horizon, action_dim]``.

    Each downsampling level halves the horizon, so ``horizon`` must be divisible
    by ``2 ** (len(down_dims) - 1)``.
    """

    def __init__(
        self,
        action_dim: int,
        global_cond_dim: int,
        down_dims: Sequence[int],
        kernel_size: int,
        n_groups: int,
        diffusion_step_embed_dim: int,
        use_film_scale_modulation: bool,
    ):
        super().__init__()
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )
        block_kwargs = {
            "cond_dim": diffusion_step_embed_dim + global_cond_dim,
            "kernel_size": kernel_size,
            "n_groups": n_groups,
            "use_film_scale_modulation": use_film_scale_modulation,
        }
        in_out = [(action_dim, down_dims[0])] + list(zip(down_dims[:-1], down_dims[1:]))

        self.down_modules = nn.ModuleList()
        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = idx == len(in_out) - 1
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1d(dim_in, dim_out, **block_kwargs),
                        ConditionalResidualBlock1d(dim_out, dim_out, **block_kwargs),
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1)
                        if not is_last
                        else nn.Identity(),
                    ]
                )
            )

        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1d(
                    down_dims[-1], down_dims[-1], **block_kwargs
                ),
                ConditionalResidualBlock1d(
                    down_dims[-1], down_dims[-1], **block_kwargs
                ),
            ]
        )

        self.up_modules = nn.ModuleList()
        for dim_out, dim_in in reversed(in_out[1:]):
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1d(dim_in * 2, dim_out, **block_kwargs),
                        ConditionalResidualBlock1d(dim_out, dim_out, **block_kwargs),
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size, n_groups),
            nn.Conv1d(down_dims[0], action_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        global_cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = x.transpose(1, 2)
        cond = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            cond = torch.cat([cond, global_cond], dim=-1)

        skips = []
        for block1, block2, downsample in self.down_modules:
            x = block2(block1(x, cond), cond)
            skips.append(x)
            x = downsample(x)
        for block in self.mid_modules:
            x = block(x, cond)
        for block1, block2, upsample in self.up_modules:
            x = torch.cat([x, skips.pop()], dim=1)
            x = upsample(block2(block1(x, cond), cond))
        return self.final_conv(x).transpose(1, 2)
