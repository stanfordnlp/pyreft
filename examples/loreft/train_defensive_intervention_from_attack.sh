export attack_file_path=$1

python train_defensive_intervention.py \
--model_name_or_path meta-llama/Llama-2-7b-chat-hf \
--layers "18;28" \
--low_rank_dimension 2 \
--n_train_examples 50 \
--batch_size 10 \
--learning_rate 4e-3 \
--num_train_epochs 5.0 \
--output_dir defense_results_2 \
--logging_steps 1 \
--positions "f1+l1" \
--share_weights \
--nonstop \
--attack_prompt_file "${attack_file_path}"
