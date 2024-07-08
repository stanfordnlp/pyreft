# The name of this experiment.
name=$2

# Save logs and models under snap/vqa; make backup.
output=$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# export PYTHONPATH=$PYTHONPATH:/local/harold/ubert/clip_vlp/CLIP

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    unbuffer python -m torch.distributed.launch --master_port=$3 --nproc_per_node=$4 src/tasks/gqa.py \
    --distributed \
    --train train,valid --valid testdev \
    --tqdm --output $output \
    --input_raw_images \
    --use_clip \
    --numWorkers 10 \
    --batchSize 2 --optim bert --lr 1e-5 --epochs 10 \
    --llayers 12 --xlayers 0 --rlayers 0 \
    --loadLXMERTQA snap/pretrained/CLIP_VL_RN50x4 \
    --visualbert_style \
    --vqa_style_transform \
    --fp16 \
    --add_zero_padding \
    --gradient_accumulation_steps 8 \
    --warmup_ratio 0.05 \
    --report_step 400 \
    --use_separate_optimizer_for_visual \
    --sgd_lr 0.001 \
    --sgd_momentum 0.0 \
    --schedule 3 \
    --use_positional_embedding \
    --pos_num 25 \
    --clip_model_name RN50x4 \
    --loss_scale 500 \
    ${@:5}  | tee $output/log.log


# bash scripts/gqa.sh 0 snap/gqa/full 9595 1 --gradient_accumulation_steps 8 --batchSize 32 --lr 5e-5 --warmup_ratio 0.05 --report_step 400 --loadLXMERTQA snap/pretrained/CLIP_VL_RN50x4_LXRT --use_separate_optimizer_for_visual --sgd_lr 0.001 --sgd_momentum 0.0 --epoch 5 --schedule 3 --use_positional_embedding --pos_num 25 --clip_model_name RN50x4 --loss_scale 500

# bash run/finetune/gqa.bash 4,5,6,7 snap/gqa/final_e20_small_lr 9595 4 --gradient_accumulation_steps 8 --batchSize 8 --lr 5e-5 --warmup_ratio 0.05 --report_step 400 --loadLXMERTQA /local/harold/ubert/clip_vlp/lxmert/snap/pretrain/clip_full_20_no_qa_50x4_new_continue_from_9/Epoch11 --use_separate_optimizer_for_visual --sgd_lr 0.001 --sgd_momentum 0.0 --epoch 5 --schedule 3 --use_positional_embedding --pos_num 25 --clip_model_name RN50x4 --loss_scale 500 

# bash run/finetune/gqa.bash 3,4,5,6 snap/gqa/freeze_50x4 9595 4 --gradient_accumulation_steps 8 --batchSize 8 --lr 5e-5 --warmup_ratio 0.05 --report_step 400 --use_separate_optimizer_for_visual --sgd_lr 0.001 --sgd_momentum 0.0 --epoch 5 --schedule 3 --use_positional_embedding --pos_num 25 --clip_model_name RN50x4 --loss_scale 500 --freeze_clip

# bash run/finetune/gqa.bash 5 snap/gqa/test 9595 1 --gradient_accumulation_steps 8 --batchSize 8 --lr 5e-5 --warmup_ratio 0.05 --report_step 400 --use_separate_optimizer_for_visual --sgd_lr 0.001 --sgd_momentum 0.0 --epoch 5 --schedule 3 --use_positional_embedding --pos_num 25 --clip_model_name RN50x4 --loss_scale 500 --test submit

# bash run/finetune/gqa.bash 4,5,6,7 snap/gqa/final_e20_RN50_large_lr 9595 4 --gradient_accumulation_steps 8 --batchSize 8 --lr 5e-5 --warmup_ratio 0.05 --report_step 400 --loadLXMERTQA /local/harold/ubert/clip_vlp/lxmert/snap/pretrain/clip_full_20_no_qa_continue_from_17/Epoch11 --use_separate_optimizer_for_visual --sgd_lr 0.001 --sgd_momentum 0.0 --epoch 5 --schedule 3 --use_positional_embedding --pos_num 25 --loss_scale 500 --lr 5e-5


# # bash run/finetune/gqa.bash 0 snap/gqa/test_rn50 9545 1 --gradient_accumulation_steps 1 --batchSize 8 --lr 5e-5 --warmup_ratio 0.05 --report_step 400 --freeze_clip --epoch 5 --schedule 3 --use_positional_embedding --pos_num 25 --loss_scale 500 --lr 3e-5 --test submit --load snap/gqa/final_e20_RN50/BEST

# # bash run/finetune/gqa.bash 3,4,5,6 snap/gqa/scratch_50x4_FU_TRUE 9595 4 --gradient_accumulation_steps 1 --batchSize 8 --lr 5e-5 --warmup_ratio 0.05 --report_step 400 --freeze_clip --epoch 5 --schedule 3 --use_positional_embedding --pos_num 25 --loss_scale 500 --lr 3e-5 --clip_model_name RN50x4