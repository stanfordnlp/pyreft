import enum
from .reft_model import ReftModel


class ReftType(str, enum.Enum):
    """
    Enum class for the different types of adapters in REFT.

    Supported REFT types:
    - LOREFT
    """

    LOREFT = "LOREFT"
    NLOREFT = "NOREFT"
    # Add yours here!


class TaskType(str, enum.Enum):
    """
    Enum class for the different types of tasks supported by REFT.

    Overview of the supported task types:
    - SEQ_CLS: Text classification.
    - CAUSAL_LM: Causal language modeling.
    """

    SEQ_CLS = "SEQ_CLS"
    CAUSAL_LM = "CAUSAL_LM"


def get_reft_model(model, reft_config, set_device=True):
    """
    Create an instance of ReFT model.
    """
    reft_model = ReftModel(reft_config, model)
    if set_device:
        reft_model.set_device(model.device)
    reft_model.disable_model_gradients()    
    return reft_model
