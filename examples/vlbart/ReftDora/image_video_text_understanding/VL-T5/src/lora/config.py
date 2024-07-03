from dataclasses import dataclass


@dataclass
class LoraConfig(object):
    lora_dim = 4
    lora_alpha = 32
    lora_dropout = 0.1
