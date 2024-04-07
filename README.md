<h1 align="center"> <p>pyreft<sub> by <a href="https://github.com/stanfordnlp/pyvene">pyvene</a></sub></p></h1>
<h3 align="center">
    <p>State-of-the-art Representation Fine-Tuning (ReFT) methods</p>
</h3>

> [!WARNING]
> **Hey hey! Corrections to the preprint:** We or members of the community have identified a few typos.

- (1) Hyperparameter settings presented in Table 5 and 6 in the Appendix should be swapped, i.e., GSM8K should be the one where we apply interventions to all layers. We release our training wandb logs in our [LoReFT](https://github.com/frankaging/pyreft/tree/main/examples/loreft) folder, check those to reproduce for now!
- (2) Wrong UltraLM citation, will correct that.
- (3) Commonsense170K is not 100 times larger than GSM8K :) (170/8).

We will update our arXiv paper on Monday (April 8th, 2024). Sorry guys! Till then, happy ReFTing!

# A _Powerful_, _Parameter-Efficient_, and _Interpretable_ way of fine-tuning
Want to try a fine-tuning method that uses a fraction of SoTA PEFT parameters count, while achieving potentially better performance? Introducing **pyreft**, a **representation fine-tuning (ReFT)** library that supports adapting internal language model representations via trainable interventions. With fewer fine-tuning parameters and more robust performance, **pyreft** can boost fine-tuning efficiency, decrease fine-tuning cost, while opening the doors to study the interpretability of adapting parameters.

**pyreft** supports

- Fine tuning any pretrained LMs on HuggingFace with ReFT

- Setting ReFT hyperparameters via configs

- Sharing the fine-tuned results easily to HuggingFace

> [!TIP]
> **Powerful and Parameter-Efficient:** Read [Our ReFT paper](https://arxiv.org/abs/2404.03592) for an introduction of representation fine-tuning (ReFT) and its performance.

> [!TIP]
> **Intepretable Finetuning:** Read [Composable ReFT](https://github.com/frankaging/pyreft/tree/main/examples/composition) for a sneak-peek of the interpretable nature of ReFT.

## Quickstart

Here is one verified conda env setup steps:

```bash
conda create --name awesome-reft python=3.10
conda activate awesome-reft
```

Then, install **pyreft** from pip+git:

```bash
pip install git+https://github.com/frankaging/pyreft.git
```

Or install **pyreft** from pip (coming soon):

```bash
pip install pyreft
```

Prepare a model for training with a ReFT method by wrapping the base model and ReFT configuration with `get_reft_model`. In the following example, we are using [`ConsreftIntervention`](https://github.com/stanfordnlp/pyreft/blob/main/pyreft/interventions.py#L85) (Constant LoReFT Intervention) which is even more parameter-efficient than the original LoReFT described in the paper:

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

With this config, yo are tuning `0.00006%` parameters, and 4,097 to be exact. Then, the `reft_model` can be used for any downstream tasks. We can see if we can do **rank-1 reft** to let the model to produce some **constant output**:

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
    tokenizer=tokenizer,
    model=model,
    inputs=["GO->"],
    outputs=[memo_sequence])

# train
training_args = transformers.TrainingArguments(
    num_train_epochs=1000.0,
    output_dir="./tmp",
    learning_rate=2e-3,
    logging_steps=50)
trainer = ReftTrainerForCausalLM(
    model=reft_model, tokenizer=tokenizer,
    args=training_args, **data_module)
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

We successfully compress the text into 4,097 parameters! We perform more rigious memorisation test like this one in [ReFT Interp](https://github.com/frankaging/pyreft/tree/main/examples/memorisation). 

You can do ReFT with any language modeling tasks or SFT. Check out our [`examples`](https://github.com/frankaging/pyreft/tree/main/examples) folder! **You can train a 7B chat-model close to ChatGPT-3.5-1103 (81.9 v.s. 86.3 Alpaca-eval scores) under 18 mins with a single A100 GPU + ReFT** by following steps here [`train.py`](https://github.com/frankaging/pyreft/blob/main/examples/loreft/train.py) training Llama-2 with the [Ultrafeedback dataset](https://arxiv.org/abs/2310.01377).

## Loading our 18 min-cooked `Loreft1k-Llama-2-7b-hf` from HuggingFace

For full tutorial, please take a look at [`chat_model.ipynb`](https://github.com/frankaging/pyreft/blob/main/examples/chat/chat_model.ipynb).

Loading the base LM first:

```py
import torch, transformers
from pyreft import (
    ReftModel,
    get_intervention_locations
)

prompt_no_input_template = """Below is an instruction that \
describes a task. Write a response that appropriately \
completes the request.

### Instruction:
%s

### Response:
"""

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name_or_path = "meta-llama/Llama-2-7b-hf"
reft_model_name_or_path = "zhengxuanzenwu/Loreft1k-Llama-2-7b-hf"
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name_or_path, model_max_length=2048, padding_side="right", use_fast=False)
tokenizer.pad_token = tokenizer.unk_token

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name_or_path, torch_dtype=torch.bfloat16, device_map=device)
```

Then, loading ReFT artifacts:

```py
reft_model = ReftModel.load(
    "zhengxuanzenwu/Loreft1k-Llama-2-7b-hf", model, from_huggingface_hub=True)
reft_model.set_device(device)
```

Start chatting with it:

```py
instruction = "Tell me about the NLP Group at Stanford University."

# tokenize and prepare the input
prompt = prompt_no_input_template % instruction
prompt = tokenizer(prompt, return_tensors="pt").to(device)
intervention_locations = torch.tensor([get_intervention_locations(
    last_position=prompt["input_ids"].shape[-1], positions="f5+l5",
    num_interventions=len(reft_model.interventions))]).permute(1, 0, 2).tolist()

# generate
_, reft_response = reft_model.generate(
    prompt, 
    unit_locations={"sources->base": (None, intervention_locations)},
    intervene_on_prompt=True, max_new_tokens=512, do_sample=False, 
    no_repeat_ngram_size=5, repetition_penalty=1.1,
    eos_token_id=tokenizer.eos_token_id, early_stopping=True
)
print(tokenizer.decode(reft_response[0], skip_special_tokens=True))
```
Note that Llama-2 models can follow instructions zero-shot. We encourge people to try on other more primitive base LMs and see if ReFT can work well!

**Usage and License Notices**: Our chat-model is intended and licensed for research use only. The model is CC BY NC 4.0 (allowing only non-commercial use) should not be used outside of research purposes. 


## Why you should use ReFT as oppose to PEFT?

There are various benefits such as saving memory and storage. In addition to that, ReFT is more interpretable and extensible than PEFT. The interventions we are learning is simply a causal abstraction of the task you are training without touching any model weights. The intervention site search space is large, and can be at any token position which is more flexibile. We showcase ReFT performance on various benchmarks against popular PEFT such as LoRA and its newer variants (e.g., DoRA) in our paper.

## Learn more through examples

| Example | Description |
|-|-|
| [pyvene](https://github.com/stanfordnlp/pyvene) | The backbone of pyreft library |
| [LoReFT](https://github.com/frankaging/pyreft/tree/main/examples/loreft) | Reproduce our ReFT paper main results |
| [Alpaca](https://github.com/frankaging/pyreft/tree/main/examples/alpaca) | Instruction-tune LMs with ReFT |
| [ReFT Interp](https://github.com/frankaging/pyreft/tree/main/examples/memorisation) | Some hints on why ReFT works |
| [Composable ReFT](https://github.com/frankaging/pyreft/tree/main/examples/composition) | Some why ReFT is an interpretable method |

## Citation
Make sure you cite the **ReFT** paper:
```bibtex
@article{wuandarora2024reft,
  title={ReFT: Representation Finetuning for Language Models},
  author={Wu, Zhengxuan* and Arora, Aryaman* and Wang, Zheng and Geiger, Atticus and Jurafsky, Dan and Manning, Christopher D. and Potts, Christopher},
  booktitle={arXiv:2404.03592},
  url={arxiv.org/abs/2404.03592},
  year={2024}
}
```

And please cite the **pyvene** library paper as well:
```bibtex
@article{wu2024pyvene,
  title={pyvene: A Library for Understanding and Improving {P}y{T}orch Models via Interventions},
  author={Wu, Zhengxuan and Geiger, Atticus and Arora, Aryaman and Huang, Jing and Wang, Zheng and Goodman, Noah D. and Manning, Christopher D. and Potts, Christopher},
  booktitle={arXiv:2403.07809},
  url={arxiv.org/abs/2403.07809},
  year={2024}
}
```

## Outreach
If you are interested in integrating this library into your workflow or in reimplementing it for improved efficiency, please feel free to contact us! We may have additional insights to share.



