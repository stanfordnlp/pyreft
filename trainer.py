import pyvene as pv
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, AutoTokenizer
from datasets import Dataset
from tqdm import tqdm
import os
import torch
import re
import evaluate
import numpy as np
from sklearn.metrics import classification_report
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


device = "cuda" if torch.cuda.is_available() else "cpu"


def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    pred_answer = float(pred[-1])
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer


def extract_answer_letter(sentence: str) -> str:
    sentence_ = sentence.strip()
    pred_answers = re.findall(r'A|B|C|D|E', sentence_)
    if pred_answers:
        if not pred_answers:
            return ''
        return pred_answers[-1]
    else:
        return ''
        
        
def extract_output(pred, trigger=''):
    if not trigger:
        return pred
    # for causallm only, use special trigger to detect new tokens.
    # if cannot find trigger --> generation is too long; default to empty generation
    start = pred.find(trigger)
    if start < 0:
        return ''
    output = pred[start+len(trigger):].lstrip() # left strip any whitespaces
    return output


def make_data_collator(tokenizer, model) -> DataCollatorForSeq2Seq:
    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding="longest",
        max_length=2048,
    )


def make_dataloader(dataset: Dataset, batch_size: int, collate_fn: DataCollatorForSeq2Seq, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=collate_fn)


class ReftTrainer(Trainer):
    def get_train_dataloader(self) -> DataLoader:
        return make_dataloader(self.train_dataset, self._train_batch_size, self.data_collator, shuffle=True)


    def compute_loss(
        self,
        intervenable: pv.IntervenableModel,
        inputs,
        return_outputs=False
    ):
        
        # run intervened forward pass
        _, cf_outputs = intervenable(
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            },
            unit_locations={"sources->base": (
                None,
                inputs["intervention_locations"].permute(1, 0, 2).tolist()
            )}
        )

        # lm loss on counterfactual labels
        lm_logits = cf_outputs.logits
        labels = inputs["labels"]
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # return
        return (loss, cf_outputs) if return_outputs else loss


    def save_model(self, output_dir, _internal_call):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.model.save(save_directory=f"{output_dir}/intervenable_model")


class ReftTrainerForSequenceClassification(ReftTrainer):
    
    def compute_loss(
        self,
        intervenable: pv.IntervenableModel,
        inputs,
        return_outputs=False
    ):
        # run intervened forward pass
        _, cf_outputs = intervenable(
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            },
            unit_locations={"sources->base": (
                None,
                inputs["intervention_locations"].permute(1, 0, 2).tolist()
            )}
        )
        # classification loss on counterfactual labels
        logits = cf_outputs.logits
        labels = inputs["labels"]

        if self.model.model.config.problem_type is None:
            if self.model.model.num_labels == 1:
                problem_type = "regression"
            elif self.model.model.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                problem_type = "single_label_classification"
            else:
                problem_type = "multi_label_classification"

        if problem_type == "regression":
            loss_fct = MSELoss()
            if self.model.model.num_labels == 1:
                loss = loss_fct(logits.squeeze(), labels.squeeze().to(torch.bfloat16))
            else:
                loss = loss_fct(logits, labels.to(torch.bfloat16))
        elif problem_type == "single_label_classification":
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.model.num_labels), labels.view(-1))
        elif problem_type == "multi_label_classification":
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        # return
        return (loss, cf_outputs) if return_outputs else loss


def compute_metrics(
    task: str,
    dataset_name: str,
    intervenable: pv.IntervenableModel,
    tokenizer: AutoTokenizer,
    eval_dataset: Dataset,
    data_items: list,
    trigger_tokens: str,
    run_name: str,
    batch_size: int=4,
    data_collator=None,
    split=None,
):
    # switch the tokenizer mode first for generation tasks
    if task != "glue":
        tokenizer.padding_side = "left" # switch padding side for collator
        num_beams = 4 if task in ["commonsense", "math"] else 1

    data_collator = data_collator if data_collator is not None else \
        make_data_collator(tokenizer, intervenable.model)
    eval_dataloader = make_dataloader(eval_dataset, batch_size, data_collator, shuffle=False)
    correct_count = 0
    total_count = 0
    generations = []
    eval_iterator = tqdm(eval_dataloader, position=0, leave=True)
    all_preds = []
    all_labels = []
    
    for step, inputs in enumerate(eval_iterator):
        for k, v in inputs.items():
            if v is not None and isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)
        
        # [layers, batch_size, positions]
        intervention_locations = inputs["intervention_locations"].permute(1, 0, 2).to(device)

        if task == "glue":

            _, cf_outputs = intervenable(
                {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
                unit_locations={"sources->base": (None, intervention_locations)})
        
            # lm loss on counterfactual labels
            if dataset_name != "stsb":
                preds = cf_outputs.logits.argmax(dim=-1)
            else:
                preds = cf_outputs.logits.squeeze(dim=1)
            
            labels = inputs["labels"]
            all_preds += preds.tolist()
            all_labels += labels.tolist()
        
        else:
            # get left padding count, [batch_size], and add to locations
            left_padding = (inputs["input_ids"] == tokenizer.bos_token_id).nonzero(as_tuple=True)[1]
            left_padding = left_padding.reshape(1, -1, 1).to(device) # [1, batch_size, 1]
            intervention_locations += left_padding
            
            # for i in range(inputs["input_ids"].shape[0]):
            #     print("batch num", i)
            #     for j in range(inputs["input_ids"].shape[1]):
            #         tok = pv.models.basic_utils.format_token(tokenizer, inputs["input_ids"][i, j])
            #         print(f"{tok:<20}", end='')
            #         if intervention_locations[0, i, 0] == j:
            #             print("<-- FIRST")
            #         elif intervention_locations[-1, i, 0] == j:
            #             print("<-- LAST")
            #         else:
            #             print()
            #     input()
    
            # repeat each batch by num_beams times in intervention locations
            # -> [layers, batch_size * num_beams, positions]
            intervention_locations = intervention_locations.repeat_interleave(num_beams, dim=1).tolist()
            
            # set generation args depending on task
            generation_args = {
                "base": {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
                "unit_locations": {"sources->base": (None, intervention_locations)},
                "intervene_on_prompt": True,
                "eos_token_id": tokenizer.eos_token_id,
                "early_stopping": True,
            }
            if task == "commonsense":
                # align with https://github.com/AGI-Edgerunners/LLM-Adapters
                generation_args["max_new_tokens"] = 32
                generation_args["temperature"] = 0.1
                generation_args["top_p"] = 0.75
                generation_args["top_k"] = 40
                generation_args["num_beams"] = num_beams
                generation_args["do_sample"] = True
            elif task == "math":
                # slightly changed to optimize our performance on top of
                # https://github.com/AGI-Edgerunners/LLM-Adapters
                generation_args["max_new_tokens"] = 256
                generation_args["temperature"] = 0.3
                generation_args["top_p"] = 0.75
                generation_args["top_k"] = 40
                generation_args["num_beams"] = num_beams
                generation_args["do_sample"] = True
            elif task in ["alpaca", "instruct", "ultrafeedback"]:
                generation_args["max_length"] = 2048
                # align with https://arxiv.org/abs/2402.15179
                generation_args["no_repeat_ngram_size"] = 5
                generation_args["repetition_penalty"] = 1.1
                generation_args["do_sample"] = False
    
            # generate with intervention on prompt
            _, steered_response = intervenable.generate(**generation_args)
    
            # detokenize in batch
            actual_preds = tokenizer.batch_decode(steered_response, skip_special_tokens=True)
            
            for id, pred in zip(inputs["id"].tolist(), actual_preds):
                example = data_items[id]
                try:
                    raw_generation = extract_output(pred, trigger_tokens)
                except:
                    print("get not split based on trigger tokens: ", raw_generation)
                    raw_generation = "WRONG"
    
                # check if generation is correct
                if task == "commonsense":
                    answer = example["answer"]
                    generation = raw_generation[:]
                    if generation.strip() == answer.strip():
                        correct_count += 1
                elif task == "math":
                    answer = example["answer"]
                    answer = answer.strip()
                    if dataset_name == "AQuA":
                        generation = extract_answer_letter(raw_generation)
                        if generation.strip() == answer.strip():
                            correct_count += 1
                    else:
                        generation = extract_answer_number(raw_generation)
                        if abs(float(answer) - generation) <= 0.001:
                            correct_count += 1
                
                # log
                total_count += 1
                if task not in ["alpaca", "instruct", "ultrafeedback"]:
                    metric_str = round(correct_count / total_count, 3)
                    eval_iterator.set_postfix({"em": metric_str})
                    generations += [{
                        "instruction": example["instruction"],
                        "raw_generation": pred,
                        "generation": generation,
                        "answer": answer
                    }]
                else:
                    generations += [{
                        "instruction": example["instruction"],
                        "raw_generation": pred,
                        "output": raw_generation,
                        "dataset": dataset_name,
                        "generator": run_name
                    }]
    # compute metrics
    if task == "glue":
        metric = evaluate.load("glue", dataset_name)
        def compute_metrics_glue(preds, labels):
            result = metric.compute(predictions=preds, references=labels)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        
        # print(classification_report(all_labels, all_preds, digits=3))
        report = compute_metrics_glue(all_labels, all_preds)
        print_str = "task metrics "
        if split:
            report = {split + "_" + k: v for k, v in report.items()}
            print_str += "[" + split + "]"
        print_str += ":"
        print(report)
        return [], report
    if task in ["alpaca", "instruct", "ultrafeedback"]:
        return generations, {}
    else:
        return generations, {f"eval/{dataset_name}": correct_count / total_count}
