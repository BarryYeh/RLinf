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
from rlinf.models.embodiment.diffusion_policy.scheduler import DDPMScheduler
from rlinf.models.embodiment.multitask_dit import modules
from rlinf.models.embodiment.multitask_dit.modules import DiffusionTransformer

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass
class MultitaskDiTConfig:
    """Language-conditioned diffusion transformer, following LeRobot's ``MultiTaskDiTPolicy``.

    The policy sees ``n_obs_steps`` main-camera frames spaced ``num_action_chunks``
    env steps apart and the task instruction, predicts ``horizon`` actions and
    executes the first ``num_action_chunks`` of them. It has no proprio state, so
    it runs unchanged in LIBERO and in the world model.
    """

    action_dim: int = 7
    num_action_chunks: int = 8
    horizon: int = 32
    n_obs_steps: int = 2
    action_min: list[float] = field(default_factory=lambda: [-1.0] * 7)
    action_max: list[float] = field(default_factory=lambda: [1.0] * 7)

    resize_shape: Optional[list[int]] = None
    crop_shape: list[int] = field(default_factory=lambda: [224, 224])
    vision_encoder_name: str = "openai/clip-vit-base-patch16"
    text_encoder_name: str = "openai/clip-vit-base-patch16"
    vision_encoder_lr_multiplier: float = 0.1

    hidden_dim: int = 768
    num_layers: int = 8
    num_heads: int = 12
    dropout: float = 0.1
    timestep_embed_dim: int = 256
    rope_base: float = 10000.0

    num_train_timesteps: int = 100
    num_inference_steps: int = 100
    clip_sample_range: float = 1.0

    # Derived by rlinf.config for evaluation: rollout calls per episode.
    obs_history_reset_interval: Optional[int] = None

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "MultitaskDiTConfig":
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
        if self.hidden_dim % self.num_heads != 0 or (
            self.hidden_dim // self.num_heads % 2 != 0
        ):
            raise ValueError(
                f"Expected hidden_dim divisible by num_heads into an even head size "
                f"for RoPE, got hidden_dim={self.hidden_dim}, "
                f"num_heads={self.num_heads}."
            )
        for name in ("action_min", "action_max"):
            if len(getattr(self, name)) != self.action_dim:
                raise ValueError(
                    f"Expected {name} of length action_dim={self.action_dim}, "
                    f"got {len(getattr(self, name))}."
                )
        if self.resize_shape is not None and any(
            c > s for c, s in zip(self.crop_shape, self.resize_shape)
        ):
            raise ValueError(
                f"Expected crop_shape {self.crop_shape} to fit in resize_shape "
                f"{self.resize_shape}."
            )


class MultitaskDiTPolicy(nn.Module, BasePolicy):
    """DDPM action diffusion conditioned on CLIP features of frames and instruction.

    Training batches carry ``images`` (uint8 ``[B, n_obs_steps, H, W, 3]``),
    ``task`` (``B`` instruction strings) and raw ``actions``
    (``[B, horizon, action_dim]``). The CLIP vision encoder trains at
    ``vision_encoder_lr_multiplier`` times the base learning rate; the CLIP text
    encoder is frozen and only its projection trains. Rollout keeps a per-slot frame
    history with the same layout rules as ``DiffusionPolicy``.
    """

    def __init__(self, cfg: MultitaskDiTConfig):
        super().__init__()
        self.cfg = cfg
        self.vision_encoder, self.text_encoder, self.tokenizer = (
            modules.load_clip_encoders(cfg.vision_encoder_name, cfg.text_encoder_name)
        )
        image_size = self.vision_encoder.config.image_size
        if list(cfg.crop_shape) != [image_size, image_size]:
            raise ValueError(
                f"Expected crop_shape [{image_size}, {image_size}], the input size of "
                f"{cfg.vision_encoder_name}, got {list(cfg.crop_shape)}."
            )
        self.text_encoder.requires_grad_(False)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, cfg.hidden_dim)
        vision_dim = self.vision_encoder.config.hidden_size
        self.dit = DiffusionTransformer(
            action_dim=cfg.action_dim,
            horizon=cfg.horizon,
            cond_dim=vision_dim * cfg.n_obs_steps + cfg.hidden_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            timestep_embed_dim=cfg.timestep_embed_dim,
            rope_base=cfg.rope_base,
        )
        self.noise_scheduler = DDPMScheduler(
            cfg.num_train_timesteps, cfg.clip_sample_range
        )
        self.register_buffer("action_min", torch.tensor(cfg.action_min))
        self.register_buffer("action_max", torch.tensor(cfg.action_max))
        self.register_buffer(
            "image_mean", torch.tensor(_CLIP_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "image_std", torch.tensor(_CLIP_STD).view(1, 3, 1, 1), persistent=False
        )
        # Read by the FSDP optimizer builder; matched against parameter names.
        self.lr_multipliers = {"vision_encoder.": cfg.vision_encoder_lr_multiplier}
        self._history = FrameHistory(cfg.n_obs_steps, cfg.obs_history_reset_interval)

    @property
    def num_action_chunks(self) -> int:
        return self.cfg.num_action_chunks

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        scale = (self.action_max - self.action_min).clamp(min=1e-8)
        return (actions - self.action_min) / scale * 2 - 1

    def unnormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return (actions + 1) / 2 * (self.action_max - self.action_min) + self.action_min

    def _crop(self, x: torch.Tensor) -> torch.Tensor:
        """Crop ``[B, S, C, H, W]``, one random window per sample in training."""
        crop_h, crop_w = self.cfg.crop_shape
        height, width = x.shape[-2:]
        if not self.training:
            top, left = (height - crop_h) // 2, (width - crop_w) // 2
            return x[..., top : top + crop_h, left : left + crop_w]
        tops = torch.randint(0, height - crop_h + 1, (x.shape[0],)).tolist()
        lefts = torch.randint(0, width - crop_w + 1, (x.shape[0],)).tolist()
        return torch.stack(
            [
                sample[..., top : top + crop_h, left : left + crop_w]
                for sample, top, left in zip(x, tops, lefts)
            ]
        )

    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """``[B, S, H, W, 3]`` uint8 to the CLS features ``[B, S * D]``."""
        batch_size, num_steps = images.shape[:2]
        x = images.to(self.action_min.device).permute(0, 1, 4, 2, 3).float() / 255.0
        if self.cfg.resize_shape is not None:
            x = F.interpolate(
                x.flatten(0, 1),
                size=tuple(self.cfg.resize_shape),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).unflatten(0, (batch_size, num_steps))
        x = (self._crop(x).flatten(0, 1) - self.image_mean) / self.image_std
        # LeRobot conditions on the CLS token before the final layer norm.
        cls = self.vision_encoder(pixel_values=x).last_hidden_state[:, 0]
        return cls.reshape(batch_size, -1)

    def _encode_tasks(self, tasks: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(
            list(tasks), padding=True, truncation=True, return_tensors="pt"
        ).to(self.action_min.device)
        with torch.no_grad():
            features = self.text_encoder(**tokens).pooler_output
        return self.text_proj(features)

    def _encode_obs(self, images: torch.Tensor, tasks: list[str]) -> torch.Tensor:
        if len(tasks) != images.shape[0]:
            raise ValueError(
                f"Expected one instruction per sample, got {len(tasks)} for "
                f"{images.shape[0]} samples."
            )
        image_features = self._encode_images(images)
        task_features = self._encode_tasks(tasks).to(image_features.dtype)
        return torch.cat([image_features, task_features], dim=-1)

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        return super().forward(forward_type=forward_type, **kwargs)

    def sft_forward(self, data: dict[str, Any]) -> dict[str, torch.Tensor]:
        cond = self._encode_obs(data["images"], data["task"])
        actions = self.normalize_actions(data["actions"].to(cond.device).float())
        noise = torch.randn_like(actions)
        timesteps = torch.randint(
            0,
            self.cfg.num_train_timesteps,
            (actions.shape[0],),
            device=actions.device,
        )
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
        eps = self.dit(noisy_actions, timesteps, cond)
        return {"loss": F.mse_loss(eps.float(), noise)}

    def default_forward(self, **kwargs):
        raise NotImplementedError(
            "MultitaskDiTPolicy has no RL training forward; train it with "
            "ForwardType.SFT."
        )

    @torch.no_grad()
    def sample_actions(
        self,
        images: torch.Tensor,
        tasks: list[str],
        generator: Optional[torch.Generator] = None,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """Denoise full ``horizon``s of raw actions from frames and instructions.

        Returns ``[B * num_samples, horizon, action_dim]``; the ``num_samples``
        draws for one input are adjacent, and the observation is encoded once.
        """
        cond = self._encode_obs(images, tasks).repeat_interleave(num_samples, dim=0)
        sample = torch.randn(
            (cond.shape[0], self.cfg.horizon, self.cfg.action_dim),
            generator=generator,
            device=cond.device,
        )
        num_steps = self.cfg.num_inference_steps
        for t in self.noise_scheduler.timesteps(num_steps):
            timesteps = torch.full(
                (sample.shape[0],), t, device=sample.device, dtype=torch.long
            )
            eps = self.dit(sample, timesteps, cond).float()
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
            env_obs: RLinf env observations; ``main_images`` (uint8
                ``[B, H, W, 3]``) and ``task_descriptions`` (``B`` strings) are read.
            mode: Rollout mode. Only ``"eval"`` is supported.
            **kwargs: Ignored sampling parameters passed by the rollout worker.
        """
        del kwargs
        if mode != "eval":
            raise NotImplementedError(
                f"MultitaskDiTPolicy rollout supports mode='eval' only, got {mode!r}."
            )
        images = self._history.update(torch.as_tensor(env_obs["main_images"]))
        tasks = env_obs["task_descriptions"]
        actions = self.sample_actions(images, tasks)[:, : self.cfg.num_action_chunks]
        actions = actions.cpu()
        placeholder = torch.zeros(actions.shape[:2])
        result = {
            "prev_logprobs": placeholder,
            "prev_values": placeholder.clone(),
            "forward_inputs": {"action": actions},
        }
        return actions, result
