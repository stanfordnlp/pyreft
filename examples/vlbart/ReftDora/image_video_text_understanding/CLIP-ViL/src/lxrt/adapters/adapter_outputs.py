"""Defines the output class for the adapter layers' parameters."""
import torch
from dataclasses import dataclass


@dataclass
class SamplerOutput:
    """Base class for the base and weights of each adapter."""
    weight: torch.FloatTensor = None
    bias: torch.FloatTensor = None


@dataclass
class LayerNormOutput:
    """Base class for the base and weights of the conditional
    layer norms."""
    weight: torch.FloatTensor = None
    bias: torch.FloatTensor = None


@dataclass
class AdapterOutput:
    """Base class for each adapter weights"""
    up: SamplerOutput = None
    down: SamplerOutput = None
    pre_norm: LayerNormOutput = None
    post_norm: LayerNormOutput = None


@dataclass
class AdapterT5BlockOutput:
    """
    Base class for adapter layer's outputs.
    """
    feed_forward: AdapterOutput = None
    self_attention: AdapterOutput = None
    cross_attention: AdapterOutput = None