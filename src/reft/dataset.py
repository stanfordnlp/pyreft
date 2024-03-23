IGNORE_INDEX = -100

no_header_prompt_template = """\
### Instruction:
%s

### Response:
"""

prompt_input = """Below is an instruction that \
describes a task, paired with an input that provides \
further context. Write a response that appropriately \
completes the request.

### Instruction:
%s

### Input:
%s

### Response:
"""

prompt_no_input = """Below is an instruction that \
describes a task. Write a response that appropriately \
completes the request.

### Instruction:
%s

### Response:
"""

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset


from abc import abstractmethod

class ReftDataset(Dataset):

    def get_intervention_locations(self, **kwargs):
        """
        This function generates the intervention locations.

        For your customized dataset, you want to create your own function.
        """
        # parse kwargs
        share_weights = kwargs["share_weights"]
        last_position = kwargs["last_position"]
        _first_n, _last_n = kwargs["first_n"], kwargs["last_n"]
        num_interventions = kwargs["num_interventions"]
        pad_mode = kwargs["pad_mode"]

        first_n = min(last_position // 2, _first_n)
        last_n = min(last_position // 2, _last_n)
    
        pad_amount = (_first_n - first_n) + (_last_n - last_n)
        pad_position = -1 if pad_mode == "first" else last_position
        if share_weights or (first_n == 0 or last_n == 0):
            position_list = [i for i in range(first_n)] + \
                [i for i in range(last_position - last_n, last_position)] + \
                [pad_position for _ in range(pad_amount)]
            intervention_locations = [position_list]*num_interventions
        else:
            left_pad_amount = (_first_n - first_n)
            right_pad_amount = (_last_n - last_n)
            left_intervention_locations = [i for i in range(first_n)] + [pad_position for _ in range(left_pad_amount)]
            right_intervention_locations = [i for i in range(last_position - last_n, last_position)] + \
                [pad_position for _ in range(right_pad_amount)]
            # after padding, there could be still length diff, we need to do another check
            left_len = len(left_intervention_locations)
            right_len = len(right_intervention_locations)
            if left_len > right_len:
                right_intervention_locations += [pad_position for _ in range(left_len-right_len)]
            else:
                left_intervention_locations += [pad_position for _ in range(right_len-left_len)]
            intervention_locations = [left_intervention_locations]*(num_interventions//2) + \
                [right_intervention_locations]*(num_interventions//2)
        
        return intervention_locations

