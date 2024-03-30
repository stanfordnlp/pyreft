<h1 align="center"> <p>pyReFT<sub> by <a href="https://github.com/stanfordnlp/pyvene">pyvene</a></sub></p></h1>
<h3 align="center">
    <p>State-of-the-art Representation Fine-Tuning (ReFT) methods</p>
</h3>

# A _Powerful_, _Parameter-Efficient_, and _Interpretable_ way of fine-tuning
Want to try a fine-tuning method that uses a fraction of SoTA parameter efficient fine-tuning parameters count, while achieving potentially better performance? Introducing **pyReFT**, a **representation fine-tuning (ReFT)** library that supports adapting internal language model representations via trainable interventions. With fewer fine-tuning parameters and more robust performance, **pyReFT** can boost fine-tuning efficiency, decrease fine-tuning cost, while opening the doors to study the interpretability of adapting parameters.

**pyReFT** supports

- Fine tuning any pretrained LMs on HuggingFace with ReFT

- Setting ReFT hyperparameters via configs

- Sharing the fine-tuned results easily to HuggingFace

> [!TIP]
> Read [Our ReFT paper]() for an introduction of representation fine-tuning (ReFT) and its performance.

## Quickstart

Install **pyReFT** from pip+git:

```bash
pip install git+https://github.com/frankaging/pyreft.git
```

Or install **pyReFT** from pip (coming soon):

```bash
pip install pyreft
```

Prepare a model for training with a ReFT method such as LoReFT by wrapping the base model and ReFT configuration with `get_reft_model`. With ReFT, you are only tuning **0.0001%** of the model's original parameters! (Try it! You can instruction-tune that **0.0001%**, and have a pretty good chat-model!)

```python
import torch
import transformers

from pyreft import (
    get_reft_model,
    ReftConfig,
    ConsreftIntervention
)

# loading huggingface model
model_name_or_path = "yahma/llama-7b-hf"
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name_or_path, torch_dtype=torch.bfloat16, device_map="cuda")

# wrap the model with rank-1 constant reft
reft_config = ReftConfig(representations={"layer": 15, "component": "block_output",
    "intervention": ConsreftIntervention(
    embed_dim=model.config.hidden_size, low_rank_dimension=1)})
reft_model = get_reft_model(model, reft_config)
reft_model.print_trainable_parameters()

"trainable intervention params: 4,097 || trainable model params: 0"
"model params: 6,738,415,616 || trainable%: 6.080064266549391e-05"
```

Then, the `reft_model` can be used for any downstream tasks. We can see if we can do **rank-1 reft** to let the model to produce some **constant output**:

```python
from pyreft import (
    ReftTrainerForCausalLM,
    make_last_position_supervised_data_module
)
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name_or_path, model_max_length=2048, padding_side="right", use_fast=False)
tokenizer.pad_token = tokenizer.unk_token

# get training data to train our intervention to remember the following sequence
memo_sequence = """
Welcome to the Natural Language Processing Group at Stanford University!
We are a passionate, inclusive group of students and faculty, postdocs
and research engineers, who work together on algorithms that allow computers
to process, generate, and understand human languages. Our interests are very
broad, including basic scientific research on computational linguistics,
machine learning, practical applications of human language technology,
and interdisciplinary work in computational social science and cognitive
science. We also develop a wide variety of educational materials
on NLP and many tools for the community to use, including the Stanza
toolkit which processes text in over 60 human languages.
"""
data_module = make_last_position_supervised_data_module(
    tokenizer=tokenizer, model=model, ["GO->"], [memo_sequence])

# train
training_args = transformers.TrainingArguments(
    num_train_epochs=1000.0, output_dir="./tmp", learning_rate=2e-3)
trainer = ReftTrainerForCausalLM(
    model=reft_model, tokenizer=tokenizer,
    args=transformers.TrainingArguments(output_dir="./tmp"), **data_module)
_ = trainer.train()
```

Once you are done with your training, you can check your model generations:

```python
prompt = tokenizer("GO->", return_tensors="pt").to("cuda")
base_unit_location = prompt["input_ids"].shape[-1] - 1  # last position
_, reft_response = reft_model.generate(
    prompt, unit_locations={"sources->base": (None, [[[base_unit_location]]])},
    intervene_on_prompt=True, max_new_tokens=512, do_sample=False, 
    eos_token_id=tokenizer.eos_token_id, early_stopping=True
)
print(tokenizer.decode(reft_response[0], skip_special_tokens=True))

"""GO->
Welcome to the Natural Language Processing Group at Stanford University!
We are a passionate, inclusive group of students and faculty, postdocs
and research engineers, who work together on algorithms that allow computers
to process, generate, and understand human languages. Our interests are very
broad, including basic scientific research on computational linguistics,
machine learning, practical applications of human language technology,
and interdisciplinary work in computational social science and cognitive
science. We also develop a wide variety of educational materials
on NLP and many tools for the community to use, including the Stanza
toolkit which processes text in over 60 human languages."""
```


You can do ReFT with any language modeling tasks or SFT. Check out our [`examples`](https://github.com/frankaging/pyreft/tree/main/examples) folder!


## Why you should use ReFT as opppose to PEFT?

There are various benefits such as saving memory and storage. In addition to that, ReFT is more interpretable and extensible than PEFT. The interventions we are learning is simply a causal abstraction of the task you are training without touching any model weights. The intervention site search space is large, and can be at any token position which is more flexibile. We showcase ReFT performance on various benchmarks against popular PEFT such as LoRA and its newer variants (e.g., DoRA) in our paper.

## Learn more through examples

| Example | Description |
|-|-|
| [pyvene](https://github.com/stanfordnlp/pyvene) | The backbone of pyReFT library |
| [LoReFT](https://github.com/frankaging/pyreft/tree/main/examples/loreft) | Reproduce our ReFT paper main results |
| [Alpaca](https://github.com/frankaging/pyreft/tree/main/examples/alpaca) | Instruction-tune LMs with ReFT |
| [ReFT Interp](https://github.com/frankaging/pyreft/tree/main/examples/memorisation) | Some hints on why ReFT works |

## Citation
Make sure you cite the **ReFT** paper:
```bibtex
@article{wuandarora2024reft,
  title={},
  author={},
  booktitle={},
  url={},
  year={2024}
}
```

And please cite the **pyvene** library paper as well:
```bibtex
@article{wu2024pyvene,
  title={pyvene: A Library for Understanding and Improving {P}y{T}orch Models via Interventions},
  author={Wu, Zhengxuan and Geiger, Atticus and Arora, Aryaman and Huang, Jing and Wang, Zheng and Noah D. Goodman and Christopher D. Manning and Christopher Potts},
  booktitle={arXiv:2403.07809},
  url={arxiv.org/abs/2403.07809},
  year={2024}
}
```

## Outreach
If you are interested in integrating this library into your workflow or in reimplementing it for improved efficiency, please feel free to contact us! We may have additional insights to share.



