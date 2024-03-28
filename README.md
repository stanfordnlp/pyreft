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

Prepare a model for training with a PEFT method such as LoRA by wrapping the base model and PEFT configuration with `get_peft_model`. For the bigscience/mt0-large model, you're only training 0.19% of the parameters!

```python
from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
model_name_or_path = "bigscience/mt0-large"
tokenizer_name_or_path = "bigscience/mt0-large"

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
"trainable params: 2359296 || all params: 1231940608 || trainable%: 0.19151053100118282"
```

To load a PEFT model for inference:

```py
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

model = AutoPeftModelForCausalLM.from_pretrained("ybelkada/opt-350m-lora").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

model.eval()
inputs = tokenizer("Preheat the oven to 350 degrees and place the cookie dough", return_tensors="pt")

outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=50)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])

"Preheat the oven to 350 degrees and place the cookie dough in the center of the oven. In a large bowl, combine the flour, baking powder, baking soda, salt, and cinnamon. In a separate bowl, combine the egg yolks, sugar, and vanilla."
```