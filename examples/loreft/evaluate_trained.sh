export task=$1
export loaddir=$2

# NOTE: currently must replicate exact config from training script

python train.py -task $task \
-data_dir dataset \
-model "meta-llama/Llama-2-7b-chat-hf" \
-seed 42 \
-l "18;28" -r 2 -p "f1+l1" -e 1 -lr 9e-4 \
-type NoreftIntervention \
-gradient_accumulation_steps 1 \
-batch_size 16 \
-eval_batch_size 16 \
--dropout 0.00 \
--test_split test \
--use_normalized_template \
--share_weights \
--warmup_ratio 0.1 \
--greedy_decoding \
--max_n_train_example 1 \
--load_dir "${loaddir}" \
--skip_train