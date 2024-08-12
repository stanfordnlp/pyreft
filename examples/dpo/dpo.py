import os
import argparse
import pyreft
from sklearn.model_selection import train_test_split
import pandas as pd
import transformers
from datasets import Dataset
from dpo_trainer import DPOReftTrainer
import wandb
import torch
import numpy as np
import random

# set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def main(
    # data arguments
    data_file: str = 'TruthfulQA/TruthfulQA.csv',
    # model + reft arguments
    model_name_or_path: str = "meta-llama/Llama-2-7b-chat-hf",
    layers: str = "18;28",
    rank: int = 4,
    positions: str = "f1+l1",
    # training arguments
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 10,
    learning_rate: float = 1e-3,
    # dpo arguments
    beta: float = 0.1,
    max_length: int = 256,
    max_prompt_length: int = 128,
    # logging arguments
    report_to_wandb: bool = True,
    log_dir: str = "./tmp",
    logging_steps: int = 40
):

    ################################
    # load data                    #
    ################################
    assert os.path.exists(data_file), f"Data file {data_file} not found."
    df = pd.read_csv(data_file)
    df_train, df_test = train_test_split(df, train_size=0.8, random_state=SEED)

    ################################
    # load model and tokenizer     #
    ################################
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt_no_input_template = """<s>[INST] %s [/INST]"""

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=2048,
        padding_side="right",
        use_fast=False
    )
    tokenizer.pad_token = tokenizer.unk_token

    ################################
    # load reft model             #
    ################################
    layers = [int(l) for l in layers.split(";")]
    reft_config = pyreft.ReftConfig(representations=[
        {
            "layer": layer,
            "component": "block_output",
            "low_rank_dimension": rank,
            "intervention": pyreft.LoreftIntervention(
                embed_dim=model.config.hidden_size,
                low_rank_dimension=rank
            )
        }
        for layer in layers
    ])
    reft_model = pyreft.get_reft_model(model, reft_config)
    reft_model.set_device(device)
    reft_model.print_trainable_parameters()

    ################################
    # prepare data                 #
    ################################
    prompts = []
    correct_answers = []
    incorrect_answers = []

    for _, r in df_train.iterrows():
        question = r['Question']
        correct = r['Correct Answers'].split(';')
        incorrect = r['Incorrect Answers'].split(';')

        # get the same number of correct & incorrect answers
        min_length = min(len(correct), len(incorrect))
        correct, incorrect = correct[:min_length], incorrect[:min_length]

        prompts += [prompt_no_input_template % question] * min_length
        # add newline to generated answers (since that's what llama-2 seems to do)
        correct_answers += [' ' + answer.strip() for answer in correct]
        incorrect_answers += [' ' + answer.strip() for answer in incorrect]

    data_module = pyreft.make_multiple_position_supervised_data_module(
        tokenizer, model, prompts, correct_answers,
        positions=positions, share_weights=True, num_interventions=len(layers)
    )

    train_dataset = Dataset.from_dict({
        'intervention_locations': data_module['train_dataset']['intervention_locations'],
        'prompt': prompts,
        'chosen': correct_answers,
        'rejected': incorrect_answers
    })

    # want to avoid a CUDA device-side alert for out-of-bounds intervention
    assert all([i[0][1] < len(tokenizer.encode(p)) for i, p in zip(train_dataset['intervention_locations'], train_dataset['prompt'])])
    
    ################################
    # train model                  #
    ################################
    report_to = "none"
    if report_to_wandb:
        wandb.init(project="reft_dpo")
        report_to = "wandb"

    training_args = transformers.TrainingArguments(
        num_train_epochs=num_train_epochs,
        output_dir=log_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        report_to=report_to
    )

    generate_during_eval = False
    trainer = DPOReftTrainer(
        reft_model,
        reft_model, # we pass it in, but ignore the reference model during training
        args=training_args,
        beta=beta,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        tokenizer=tokenizer,
        max_length=max_length,
        max_target_length=max_length,
        max_prompt_length=max_prompt_length,
        generate_during_eval=generate_during_eval,
        peft_config=None,
    )

    trainer.train()

    ################################
    # test model                   #
    ################################
    # edit to test out custom questions
    question = "What does ADIDAS stand for?"

    # tokenize and prepare the input
    prompt = prompt_no_input_template % question
    prompt = tokenizer(prompt, return_tensors="pt").to(device)

    base_unit_location = prompt["input_ids"].shape[-1] - 1  # last position
    with torch.no_grad():
        orig_response, reft_response = reft_model.generate(
            prompt,
            unit_locations={"sources->base": (None, [[[0, base_unit_location]], [[0, base_unit_location]]])},
            intervene_on_prompt=True,
            max_new_tokens=128,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            output_original_output=True
        )

    start_idx = prompt['input_ids'].shape[-1]
    print('Question:', question)
    print('Answer (original):', tokenizer.decode(orig_response[0][start_idx:], skip_special_tokens=True))
    print('Answer (dpo+reft):', tokenizer.decode(reft_response[0][start_idx:], skip_special_tokens=True))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default='TruthfulQA/TruthfulQA.csv')
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--layers", type=str, default="18;28")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--positions", type=str, default="f1+l1")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_prompt_length", type=int, default=128)
    parser.add_argument("--report_to_wandb", type=bool, default=True)
    parser.add_argument("--log_dir", type=str, default="./tmp")
    parser.add_argument("--logging_steps", type=int, default=40)
    args = parser.parse_args()
    main(**vars(args))