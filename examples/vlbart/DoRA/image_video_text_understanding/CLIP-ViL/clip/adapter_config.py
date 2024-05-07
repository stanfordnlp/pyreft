from dataclasses import dataclass


@dataclass
class VisionAdapterConfig(object):
    """Implements the adapter configuration proposed by Houlsby et. al, 2019
    in https://arxiv.org/abs/1902.00751."""
   
    reduction_factor: int = 1
    