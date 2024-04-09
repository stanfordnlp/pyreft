# Instruction-tuning examples

Based on the script [`train.py`](https://github.com/stanfordnlp/pyreft/blob/main/examples/alpaca/train.py).

## Alpaca

You can train Alpaca (i.e., instruction-tuning LLaMA-1 7B) with ReFT using a single training script:

```bash
python train.py --model_name_or_path yahma/llama-7b-hf \
	--data_path ./alpaca_data.json \
	--output_dir ./test/ \
	--layers "8;19" \
	--rank 4 \
	--position "f1+l1" \
	--num_train_epochs 1 \
	--per_device_train_batch_size 4 \
	--per_device_eval_batch_size 4 \
	--gradient_accumulation_steps 8 \
	--evaluation_strategy "no" \
	--save_strategy "no" \
	--learning_rate 2e-5 \
	--weight_decay 0. \
	--warmup_ratio 0.03 \
	--lr_scheduler_type "cosine" \
	--logging_steps 1
```

ReFT is not only parameter-efficient, but also data-efficient. You can try to train Alpaca with 1K examples:

```bash
python train.py --model_name_or_path yahma/llama-7b-hf \
	--data_path ./alpaca_data.json \
	--output_dir ./test/ \
	--layers "8;19" \
	--rank 4 \
	--position "f1+l1" \
	--num_train_epochs 1 \
	--per_device_train_batch_size 4 \
	--per_device_eval_batch_size 4 \
	--gradient_accumulation_steps 8 \
	--evaluation_strategy "no" \
	--save_strategy "no" \
	--learning_rate 2e-5 \
	--weight_decay 0. \
	--warmup_ratio 0.03 \
	--lr_scheduler_type "cosine" \
	--logging_steps 1 \
  --max_n_train_example 1000 \
```

Training will take less than 15 mins on a single A100 (40G MEM) GPU.
