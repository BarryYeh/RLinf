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

import torch
from omegaconf import DictConfig, OmegaConf


def get_model(cfg: DictConfig, torch_dtype=torch.float32):
    """Build the policy; a non-empty ``model_path`` is a ``full_weights.pt`` to load."""
    from rlinf.models.embodiment.multitask_dit.multitask_dit import (
        MultitaskDiTConfig,
        MultitaskDiTPolicy,
    )

    model_config = MultitaskDiTConfig.from_dict(
        OmegaConf.to_container(cfg, resolve=True)
    )
    model = MultitaskDiTPolicy(model_config)
    if cfg.get("model_path"):
        model.load_state_dict(torch.load(cfg.model_path, map_location="cpu"))
    return model.to(torch_dtype)
