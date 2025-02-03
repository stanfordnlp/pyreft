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
    --max_n_train_example 1000
```

Training will take less than 15 mins on a single A100 (40G MEM) GPU.

## Multi-GPU with FSDP

You can also train Alpaca with FSDP on multi-GPUs.

```bash
torchrun --nproc_per_node=<your_n_gpu> --master_port=<your_port> train.py \
    --model_name_or_path <your_model_name> \
    --layers "8;19" \
    --rank 4 \
    --position "f1+l1" \
    --data_path <your_data> \
    --bf16 True \
    --output_dir <your_output_dir> \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp_config ./fsdp_config.json \
    --tf32 True
```

Here is an example `fsdp_config.json`:

```json
{
    "compute_environment": "LOCAL_MACHINE",
    "debug": false,
    "distributed_type": "FSDP",
    "downcast_bf16": "no",
    "fsdp_config": {
      "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
      "fsdp_backward_prefetch_policy": "BACKWARD_PRE",
      "fsdp_forward_prefetch": false,
      "fsdp_cpu_ram_efficient_loading": true,
      "fsdp_offload_params": false,
      "fsdp_sharding_strategy": "FULL_SHARD",
      "fsdp_state_dict_type": "SHARDED_STATE_DICT",
      "fsdp_sync_module_states": true,
      "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
      "fsdp_use_orig_params": true
    },
    "machine_rank": 0,
    "main_training_function": "main",
    "mixed_precision": "bf16",
    "rdzv_backend": "static",
    "same_network": true,
    "tpu_env": [],
    "tpu_use_cluster": false,
    "tpu_use_sudo": false,
    "use_cpu": false
}
```
