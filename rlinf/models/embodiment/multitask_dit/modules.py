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

"""Diffusion transformer of LeRobot's ``MultiTaskDiTPolicy`` (AdaLN-Zero, RoPE)."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from rlinf.models.embodiment.diffusion_policy.modules import SinusoidalPosEmb


def load_clip_encoders(vision_name: str, text_name: str) -> tuple[Any, Any, Any]:
    """Pretrained CLIP vision model, CLIP text model, and text tokenizer."""
    from transformers import AutoTokenizer, CLIPTextModel, CLIPVisionModel

    return (
        CLIPVisionModel.from_pretrained(vision_name),
        CLIPTextModel.from_pretrained(text_name),
        AutoTokenizer.from_pretrained(text_name),
    )


class RotaryEmbedding(nn.Module):
    """Rotary position embedding over the action sequence."""

    def __init__(self, head_dim: int, max_seq_len: int, base: float):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        freqs = torch.outer(torch.arange(max_seq_len).float(), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos()[None, None], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None], persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[2]
        cos = self.cos[:, :, :seq_len].to(q.dtype)
        sin = self.sin[:, :, :seq_len].to(q.dtype)
        return (
            q * cos + self._rotate_half(q) * sin,
            k * cos + self._rotate_half(k) * sin,
        )


class RoPEAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        max_seq_len: int,
        rope_base: float,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len, rope_base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        qkv = self.qkv_proj(x).reshape(
            batch_size, seq_len, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        q, k = self.rope(q, k)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        return self.out_proj(out.transpose(1, 2).reshape(batch_size, seq_len, -1))


class DiTBlock(nn.Module):
    """Transformer block whose norms are modulated by the condition (AdaLN-Zero)."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        cond_dim: int,
        dropout: float,
        max_seq_len: int,
        rope_base: float,
    ):
        super().__init__()
        self.attn = RoPEAttention(
            hidden_dim, num_heads, dropout, max_seq_len, rope_base
        )
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * hidden_dim))
        # Zero init makes every block start as the identity.
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = (
            self.adaln(cond).unsqueeze(1).chunk(6, dim=-1)
        )
        x = x + gate1 * self.attn(self.norm1(x) * (1 + scale1) + shift1)
        return x + gate2 * self.mlp(self.norm2(x) * (1 + scale2) + shift2)


class DiffusionTransformer(nn.Module):
    """Predicts the noise of a ``[B, horizon, action_dim]`` action sequence."""

    def __init__(
        self,
        action_dim: int,
        horizon: int,
        cond_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        timestep_embed_dim: int,
        rope_base: float,
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(timestep_embed_dim),
            nn.Linear(timestep_embed_dim, 2 * timestep_embed_dim),
            nn.GELU(),
            nn.Linear(2 * timestep_embed_dim, timestep_embed_dim),
            nn.GELU(),
        )
        self.input_proj = nn.Linear(action_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim,
                    num_heads,
                    timestep_embed_dim + cond_dim,
                    dropout,
                    horizon,
                    rope_base,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_proj = nn.Linear(hidden_dim, action_dim)

    def forward(
        self, actions: torch.Tensor, timesteps: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        cond = torch.cat([self.time_mlp(timesteps).to(cond.dtype), cond], dim=-1)
        x = self.input_proj(actions)
        for block in self.blocks:
            x = block(x, cond)
        return self.output_proj(x)
