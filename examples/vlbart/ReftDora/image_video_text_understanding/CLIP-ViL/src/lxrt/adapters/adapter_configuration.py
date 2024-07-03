"""Implements the adapters and other parameter-efficient finetuning methods' configurations."""

from collections import OrderedDict
from dataclasses import dataclass

import torch.nn as nn

@dataclass
class AdapterConfig(object):
    """Implements the adapter configuration proposed by Houlsby et. al, 2019
    in https://arxiv.org/abs/1902.00751.
    We additionally pass all the configuration of parameter-efficient finetuning
    methods with this config."""
    add_layer_norm_before_adapter: bool = False
    add_layer_norm_after_adapter: bool = True
    non_linearity: str = "swish"
    task_reduction_factor: int = 16
    add_adapter_in_feed_forward = True
    add_adapter_in_self_attention = True
    hidden_dim = 128
    task_adapter_layers_encoder = None
    task_adapter_layers_decoder = None
    task_adapter_in_decoder = True
    intrinsic_dim = 100
    normalize_intrinsic_projections = False
    # This can be either random, or fastfood.
    intrinsic_projection = "random"

    # Hypercomplex adapters parameters 
    hypercomplex_adapters = False
    hypercomplex_division = 8
    learn_phm = True
    hypercomplex_nonlinearity="glorot-uniform"
    shared_phm_rule = False 
    factorized_phm = False 
    shared_W_phm = False
    factorized_phm_rule = False 
    phm_c_init = "normal"
    phm_rank = 1
    phm_init_range=0.01

    # prefix-tuning parameters.
    prefix_dim = 100
    init_prefix_from_vocab = False 
    kronecker_prod = False  

    # BitFit configuration.
    bitfit = False

    # Low-rank adapters.
    low_rank_adapters = False
    low_rank_w_init = "glorot-uniform"
    low_rank_rank = 1


ADAPTER_CONFIG_MAPPING = OrderedDict(
    [("adapter", AdapterConfig)])


class AutoAdapterConfig(nn.Module):
    """Generic Adapter config class to instantiate different adapter configs."""

    @classmethod
    def get(cls, config_name: str):
        if config_name in ADAPTER_CONFIG_MAPPING:
            return ADAPTER_CONFIG_MAPPING[config_name]()
        raise ValueError(
            "Unrecognized adapter config type identifier: {}. Should contain one of {}"
                .format(config_name, ", ".join(ADAPTER_CONFIG_MAPPING.keys())))
