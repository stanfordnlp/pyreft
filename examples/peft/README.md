# Combining LoRA with ReFT via "one-click"

Based on the script [`reft_with_lora.ipynb`](https://github.com/stanfordnlp/pyreft/blob/main/examples/peft/reft_with_lora.ipynb).

You can wrap any `peft` model (from the 🤗 [PEFT: State-of-the-art Parameter-Efficient Fine-Tuning library](https://github.com/huggingface/peft)) as a ReFT model with a single line of code! Then, you can co-train your LoRA wights along with interventions. 

Feel free to explore how to trade some heavy LoRA wights for some lightweight interventions!
