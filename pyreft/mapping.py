from .interventions import (
    LoreftIntervention,
    NoreftIntervention
)
from .trainer import (
    ReftTrainerForCausalLM,
    ReftTrainerForSequenceClassification
)


REFT_TYPE_TO_INTERVENTION_MAPPING = {
    "LOREFT": ReftTrainerForCausalLM,
    "NOREFT": ReftTrainerForSequenceClassification
}


MODEL_TYPE_TO_REFT_TRAINER_MAPPING = {
    "SEQ_CLS": PeftModelForSequenceClassification,
    "CAUSAL_LM": PeftModelForCausalLM,
}