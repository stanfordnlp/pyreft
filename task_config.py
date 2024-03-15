from templates import *

task_config = {
    "commonsense": {
        "train_datasets": [
            "boolq", "piqa", "social_i_qa", "hellaswag", 
            "winogrande", "ARC-Easy", "ARC-Challenge", "openbookqa"
        ],
        "eval_datasets": [
            "boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Easy", "ARC-Challenge", "openbookqa"
        ],
        "task_prompt_template": "%s\n",
        "trigger_tokens": "the correct answer is ",
    },
    "math": {
        "train_datasets": [
            "math_10k"
        ],
        "eval_datasets": [
            "MultiArith", "gsm8k", "SVAMP", "mawps", "AddSub", "AQuA", "SingleEq", 
        ],
        "task_prompt_template": alpaca_prompt_no_input_template,
        "trigger_tokens": "### Response:"
    },
    "alpaca": {
        "train_datasets": ["alpaca_data_cleaned"],
        "eval_datasets": ["alpaca_eval"],
        "task_prompt_template": alpaca_prompt_template,
        "trigger_tokens": "### Response:"
    },
    "instruct": {
        "train_datasets": ["instruct"],
        "eval_datasets": ["alpaca_eval"],
        "task_prompt_template": alpaca_prompt_template,
        "trigger_tokens": "### Response:"
    },
    "ultrafeedback": {
        "train_datasets": ["ultrafeedback"],
        "eval_datasets": ["alpaca_eval"],
        "task_prompt_template": alpaca_prompt_template,
        "trigger_tokens": "### Response:"
    },
    "glue": {
        "train_datasets": None,
        "eval_datasets": None,
        "task_prompt_template": None,
        "trigger_tokens": None
    },
    "gsm8k": {
        "train_datasets": ["gsm8k"],
        "eval_datasets": ["gsm8k"],
        "task_prompt_template": alpaca_prompt_template,
        "trigger_tokens": "### Response:"
    }
}