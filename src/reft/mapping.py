from .interventions import (
    ConditionedSourceLowRankRotatedSpaceIntervention,
    ConditionedSourceLowRankIntervention
)


REFT_TYPE_TO_INTERVENTION_MAPPING = {
    "LOREFT": ConditionedSourceLowRankIntervention,
    "NLOREFT": ConditionedSourceLowRankRotatedSpaceIntervention
}


MODEL_TYPE_TO_REFT_TRAINER_MAPPING = {
    "SEQ_CLS": PeftModelForSequenceClassification,
    "SEQ_2_SEQ_LM": PeftModelForSeq2SeqLM,
    "CAUSAL_LM": PeftModelForCausalLM,
    "TOKEN_CLS": PeftModelForTokenClassification,
    "QUESTION_ANS": PeftModelForQuestionAnswering,
    "FEATURE_EXTRACTION": PeftModelForFeatureExtraction,
}