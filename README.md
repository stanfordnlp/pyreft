<h1 align="center"> <p>pyReFT</p></h1>
<h3 align="center">
    <p>State-of-the-art Representation Fine-Tuning (ReFT) methods</p>
</h3>

# A more _Powerful_, _Parameter Efficient_, and _Interpretable_ way of fine-tuning
Want to try a fine-tuning method that uses a fraction of SoTA parameter efficient fine-tuning parameters count, while achieving potentially better performance? Introducing **PyReFT**, a **representation fine-tuning (ReFT)** library that supports adapting internal language model representations via trainable interventions. With fewer fine-tuning parameters and more robust performance, PyReFT can boost fine-tuning efficiency, decrease fine-tuning cost, while opening the doors to study the interpretability of adapting parameters.

PyReFT supports

- Fine tuning any pretrained LMs on Hugging Face with ReFT

- Setting ReFT hyperparameters via configs

- Sharing the fine-tuned results easily to Hugging Face

> [!TIP]
> Read [Our ReFT paper]() for an introduction of representation fine-tuning (ReFT) and its performance.

## Quickstart

Install PyReFT from pip:

```bash
pip install pyreft
```

Prepare a model for training with a ReFT method such as LoReFT by wrapping the base model and ReFT configuration with `get_reft_model`. With ReFT, you are only tuning **0.0001%** of the model's original parameters!

```python
from pyreft import (
    get_reft_model,
    ReftConfig
)
from pyreft.interventions import ConditionedSourceLowRankRotatedSpaceIntervention
import torch
import transformers

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name_or_path = "yahma/llama-7b-hf"

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name_or_path, torch_dtype=torch.bfloat16, device_map=device)
reft_config = ReftConfig(representations={
    "layer": 15, "component": "block_output",
    "intervention": ConditionedSourceLowRankRotatedSpaceIntervention(
    embed_dim=model.config.hidden_size, 
    low_rank_dimension=1)})
reft_model = get_reft_model(model, reft_config)
reft_model.print_trainable_parameters()
"trainable intervention params: 8,193 || trainable model params: 0"
"model params: 6,738,415,616 || trainable%: 0.00012158644504720322"
```

You can easily load a shared ReFT model for inference:

```python
model_max_length = 2048
storage_access_id = "RAND#ID1->"

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name_or_path, model_max_length=model_max_length, 
    padding_side="right", use_fast=False)
tokenizer.pad_token = tokenizer.unk_token

reft_model = reft_model.load(
    "peterwz/reft-example",
    model,
    from_huggingface_hub=True,
)
reft_model.set_device(device)

prompt = tokenizer(storage_access_id, return_tensors="pt").to(device)
base_unit_location = prompt["input_ids"].shape[-1] - 1
_, steered_response = reft_model.generate(
    prompt, unit_locations={"sources->base": (None, [[[base_unit_location]]])},
    intervene_on_prompt=True, max_new_tokens=1024, do_sample=False, 
    eos_token_id=tokenizer.eos_token_id, early_stopping=True
)
print(tokenizer.decode(steered_response[0], skip_special_tokens=True))

"RAND#ID1->Hey! This is Zhengxuan working on random stuff with LLaMA models!"
```