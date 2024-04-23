import torch
import transformers
import pyvene as pv
import argparse
from datasets import load_dataset
from pyreft import (
    ReftModel,
    ReftRewardCollator,
    ReftRewardDataset,
)
from train import (
    ReftTrainerForRewardModelling,
    compute_metrics,
    TrainingArguments,
)

# setup for gemma
pv.type_to_module_mapping[transformers.GemmaForSequenceClassification] = {
    "block_output": ("model.layers[%s]", 
                   pv.models.constants.CONST_OUTPUT_HOOK),
}
pv.type_to_dimension_mapping[transformers.GemmaForSequenceClassification] = {
    "block_output": ("hidden_size",),
}

device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.inference_mode()
def evaluate(path: str):
    # args
    model_name_or_path = "google/gemma-2b-it"
    position = "f1+l1"
    layers = "all"
    share_weights = False

    # parsing layers arg
    if layers != "all":
        layers = [int(l) for l in layers.split(";")]
    else:
        temp_config = transformers.AutoConfig.from_pretrained(model_name_or_path)
        layers = [l for l in range(temp_config.num_hidden_layers)]
    if "+" in position and not share_weights:
        layers += layers

    # load model
    reft_model_name_or_path = path
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path, model_max_length=2048, padding_side="right", use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, num_labels=1, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()

    # reft wrapping with intervention weights
    reft_model = ReftModel.load(
        reft_model_name_or_path, model)
    reft_model.set_device(device)

    # field setup
    fields = {
        "conv_A_field": "chosen", "conv_B_field": "rejected",
        "prompt_field": "prompt"
    }

    # load rewardbench
    dataset = load_dataset("allenai/reward-bench", split="train")
    subsets = set(dataset["subset"])
    results = {}
    for subset in sorted(list(subsets)):
        filtered_dataset = dataset.filter(lambda x: x["subset"] == subset)
        eval_dataset = ReftRewardDataset(
            "allenai/reward-bench", None, tokenizer,
            data_split="train",
            dataset=filtered_dataset,
            **{"num_interventions": len(layers),
            "position": position, 
            "share_weights": share_weights},
            **fields,
        )
        data_collator = ReftRewardCollator(
            tokenizer=tokenizer,
            padding=True,
            max_length=tokenizer.model_max_length
        )

        # evaluation
        trainer = ReftTrainerForRewardModelling(
            model=reft_model,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            train_dataset=eval_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            args=TrainingArguments(output_dir="eval", report_to="none"),
        )
        result = trainer.evaluate()
        results[subset] = result["accuracy"]
    
    # print summary
    for k, v in results.items():
        print(f"{k}: {v}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="./gemma_sus/reward_model")
    args = parser.parse_args()
    evaluate(**vars(args))

if __name__ == "__main__":
    main()