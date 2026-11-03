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

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.diffusion_policy.frame_history import FrameHistory
from rlinf.models.embodiment.diffusion_policy.modules import (
    ConditionalUnet1d,
    RgbEncoder,
)
from rlinf.models.embodiment.diffusion_policy.scheduler import DDPMScheduler

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class DiffusionPolicyConfig:
    """Image-only diffusion policy, following LeRobot's ``DiffusionPolicy``.

    The policy sees ``n_obs_steps`` main-camera frames spaced ``num_action_chunks``
    env steps apart, predicts ``horizon`` actions and executes the first
    ``num_action_chunks`` of them.
    """

    action_dim: int = 7
    num_action_chunks: int = 8
    horizon: int = 16
    n_obs_steps: int = 2
    action_min: list[float] = field(default_factory=lambda: [-1.0] * 7)
    action_max: list[float] = field(default_factory=lambda: [1.0] * 7)

    resize_shape: Optional[list[int]] = field(default_factory=lambda: [128, 128])
    crop_shape: list[int] = field(default_factory=lambda: [116, 116])
    spatial_softmax_num_keypoints: int = 32

    down_dims: list[int] = field(default_factory=lambda: [512, 1024, 2048])
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True

    num_train_timesteps: int = 100
    num_inference_steps: int = 100
    clip_sample_range: float = 1.0

    # Derived by rlinf.config for evaluation: rollout calls per episode.
    obs_history_reset_interval: Optional[int] = None

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "DiffusionPolicyConfig":
        names = {f.name for f in dataclasses.fields(cls)}
        config = cls(**{k: v for k, v in cfg.items() if k in names})
        config.validate()
        return config

    def validate(self) -> None:
        if self.num_action_chunks > self.horizon:
            raise ValueError(
                f"Expected num_action_chunks <= horizon, got {self.num_action_chunks} "
                f"> {self.horizon}."
            )
        downsample = 2 ** (len(self.down_dims) - 1)
        if self.horizon % downsample != 0:
            raise ValueError(
                f"Expected horizon divisible by {downsample} for down_dims "
                f"{self.down_dims}, got {self.horizon}."
            )
        for name in ("action_min", "action_max"):
            if len(getattr(self, name)) != self.action_dim:
                raise ValueError(
                    f"Expected {name} of length action_dim={self.action_dim}, "
                    f"got {len(getattr(self, name))}."
                )
        image_shape = self.resize_shape
        if image_shape is not None and any(
            c > s for c, s in zip(self.crop_shape, image_shape)
        ):
            raise ValueError(
                f"Expected crop_shape {self.crop_shape} to fit in resize_shape "
                f"{image_shape}."
            )


class DiffusionPolicy(nn.Module, BasePolicy):
    """DDPM action diffusion conditioned on a short history of main-camera frames.

    Training batches carry ``images`` (uint8 ``[B, n_obs_steps, H, W, 3]``) and
    raw ``actions`` (``[B, horizon, action_dim]``). For rollout the policy keeps
    the previous chunk's frames per batch slot itself, because the env worker only
    returns the last frame of each chunk. Slots must therefore map to the same env
    across calls, and episodes must start every ``obs_history_reset_interval``
    calls; ``rlinf.config`` enforces both for evaluation.
    """

    def __init__(self, cfg: DiffusionPolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = RgbEncoder(cfg.crop_shape, cfg.spatial_softmax_num_keypoints)
        self.unet = ConditionalUnet1d(
            action_dim=cfg.action_dim,
            global_cond_dim=self.encoder.feature_dim * cfg.n_obs_steps,
            down_dims=cfg.down_dims,
            kernel_size=cfg.kernel_size,
            n_groups=cfg.n_groups,
            diffusion_step_embed_dim=cfg.diffusion_step_embed_dim,
            use_film_scale_modulation=cfg.use_film_scale_modulation,
        )
        self.noise_scheduler = DDPMScheduler(
            cfg.num_train_timesteps, cfg.clip_sample_range
        )
        self.register_buffer("action_min", torch.tensor(cfg.action_min))
        self.register_buffer("action_max", torch.tensor(cfg.action_max))
        self.register_buffer(
            "image_mean",
            torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )
        self._history = FrameHistory(cfg.n_obs_steps, cfg.obs_history_reset_interval)

    @property
    def num_action_chunks(self) -> int:
        return self.cfg.num_action_chunks

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        scale = (self.action_max - self.action_min).clamp(min=1e-8)
        return (actions - self.action_min) / scale * 2 - 1

    def unnormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return (actions + 1) / 2 * (self.action_max - self.action_min) + self.action_min

    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """``[B, S, H, W, 3]`` uint8 to the UNet's global condition ``[B, S * D]``."""
        batch_size, num_steps = images.shape[:2]
        x = images.to(self.action_min.device).permute(0, 1, 4, 2, 3).float() / 255.0
        x = x.flatten(0, 1)
        if self.cfg.resize_shape is not None:
            x = F.interpolate(
                x,
                size=tuple(self.cfg.resize_shape),
                mode="bilinear",
                align_corners=False,
            )
        x = self._crop(x.unflatten(0, (batch_size, num_steps)).flatten(1, 2))
        x = x.unflatten(1, (num_steps, 3)).flatten(0, 1)
        x = (x - self.image_mean) / self.image_std
        return self.encoder(x).reshape(batch_size, -1)

    def _crop(self, x: torch.Tensor) -> torch.Tensor:
        """Crop ``[B, C, H, W]``, one random window per sample in training."""
        crop_h, crop_w = self.cfg.crop_shape
        height, width = x.shape[-2:]
        if not self.training:
            top, left = (height - crop_h) // 2, (width - crop_w) // 2
            return x[..., top : top + crop_h, left : left + crop_w]
        tops = torch.randint(0, height - crop_h + 1, (x.shape[0],)).tolist()
        lefts = torch.randint(0, width - crop_w + 1, (x.shape[0],)).tolist()
        return torch.stack(
            [
                sample[:, top : top + crop_h, left : left + crop_w]
                for sample, top, left in zip(x, tops, lefts)
            ]
        )

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        return super().forward(forward_type=forward_type, **kwargs)

    def sft_forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        global_cond = self._encode_images(data["images"])
        actions = self.normalize_actions(data["actions"].to(global_cond.device).float())
        noise = torch.randn_like(actions)
        timesteps = torch.randint(
            0,
            self.cfg.num_train_timesteps,
            (actions.shape[0],),
            device=actions.device,
        )
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
        eps = self.unet(noisy_actions, timesteps, global_cond)
        return {"loss": F.mse_loss(eps, noise)}

    def default_forward(self, **kwargs):
        raise NotImplementedError(
            "DiffusionPolicy has no RL training forward; train it with ForwardType.SFT."
        )

    @torch.no_grad()
    def sample_actions(
        self, images: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """Denoise a full ``horizon`` of raw actions from ``[B, S, H, W, 3]`` frames."""
        global_cond = self._encode_images(images)
        sample = torch.randn(
            (images.shape[0], self.cfg.horizon, self.cfg.action_dim),
            generator=generator,
            device=global_cond.device,
        )
        num_steps = self.cfg.num_inference_steps
        for t in self.noise_scheduler.timesteps(num_steps):
            timesteps = torch.full(
                (sample.shape[0],), t, device=sample.device, dtype=torch.long
            )
            eps = self.unet(sample, timesteps, global_cond)
            sample = self.noise_scheduler.step(eps, t, sample, num_steps, generator)
        return self.unnormalize_actions(sample)

    def reset(self) -> None:
        """Forget the frame history of every batch slot."""
        self._history.reset()

    def predict_action_batch(
        self, env_obs: dict[str, Any], mode: str = "eval", **kwargs
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return ``[B, num_action_chunks, action_dim]`` actions for env observations.

        Args:
            env_obs: RLinf env observations; only ``main_images``
                (uint8 ``[B, H, W, 3]``) is read.
            mode: Rollout mode. Only ``"eval"`` is supported.
            **kwargs: Ignored sampling parameters passed by the rollout worker.
        """
        del kwargs
        if mode != "eval":
            raise NotImplementedError(
                f"DiffusionPolicy rollout supports mode='eval' only, got {mode!r}."
            )
        images = self._history.update(torch.as_tensor(env_obs["main_images"]))
        actions = self.sample_actions(images)[:, : self.cfg.num_action_chunks]
        actions = actions.cpu()
        placeholder = torch.zeros(actions.shape[:2])
        result = {
            "prev_logprobs": placeholder,
            "prev_values": placeholder.clone(),
            "forward_inputs": {"action": actions},
        }
        return actions, result
