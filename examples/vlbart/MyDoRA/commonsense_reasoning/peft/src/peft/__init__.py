# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# coding=utf-8
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__version__ = "0.3.0.dev0"

from .mapping import MODEL_TYPE_TO_PEFT_MODEL_MAPPING, PEFT_TYPE_TO_CONFIG_MAPPING, get_peft_config, get_peft_model
from .peft_model import (
    PeftModel,
    PeftModelForCausalLM,
    PeftModelForSeq2SeqLM,
    PeftModelForSequenceClassification,
    PeftModelForTokenClassification,
)
from .tuners import (
    LoraConfig,
    LoraModel,
    DoraConfig,
    DoraModel,
    BottleneckConfig,
    BottleneckModel,
    PrefixEncoder,
    PrefixTuningConfig,
    PromptEmbedding,
    PromptEncoder,
    PromptEncoderConfig,
    PromptEncoderReparameterizationType,
    PromptTuningConfig,
    PromptTuningInit,
)
from .utils import (
    TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING,
    PeftConfig,
    PeftType,
    PromptLearningConfig,
    TaskType,
    bloom_model_postprocess_past_key_value,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    shift_tokens_right,
)
