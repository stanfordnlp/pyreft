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
    unbuffer python -m torch.distributed.launch --master_port=$3 --nproc_per_node=$4 src/tasks/snli.py \
    --distributed \
    --train train --valid valid  \
    --tqdm --output $output \
    --input_raw_images \
    --use_clip \
    --numWorkers 10 \
    --batchSize 2 --optim bert --lr 1e-5 --epochs 10 \
    --llayers 12 --xlayers 0 --rlayers 0 \
    --visualbert_style \
    --vqa_style_transform \
    --clip_model_name RN50x4 \
    --loadLXMERT snap/pretrained/CLIP_VL_RN50x4 \
    --fp16 \
    --use_adapter \
    --reduction_factor 4 \
    --add_zero_padding \
    --gradient_accumulation_steps 8 \
    --report_step 400 \
    --warmup_ratio 0.05 \
    --use_separate_optimizer_for_visual \
    --sgd_lr 0.001 \
    --sgd_momentum 0.0 \
    --schedule 1 \
    --use_positional_embedding \
    --pos_num 25 \
    --clip_model_name RN50x4 \
    ${@:5}  | tee $output/log.log


    
#bash run/finetune/snli_ve.bash 5 snap/snli/test 9595 1 --gradient_accumulation_steps 1 --batchSize 12 --lr 5e-5 --freeze_clip --loss_scale 500 --warmup_ratio 0.05 

# bash run/finetune/snli_ve.bash 4,5,6,7 snap/snli/final_e20_schedule_2 9595 4 --gradient_accumulation_steps 8 --batchSize 8 --lr 5e-5 --warmup_ratio 0.05 --report_step 400 --loadLXMERT /local/harold/ubert/clip_vlp/lxmert/snap/pretrain/clip_full_20_no_qa_50x4_new_continue_from_9/Epoch11 --use_separate_optimizer_for_visual --sgd_lr 0.001 --sgd_momentum 0.0 --epoch 2 --schedule 1 --use_positional_embedding --pos_num 25 --clip_model_name RN50x4

# bash run/finetune/snli_ve.bash 4,5,6,7 snap/snli/final_e20_RN50_schedule_2 9595 4 --gradient_accumulation_steps 8 --batchSize 8 --lr 5e-5 --warmup_ratio 0.05 --report_step 400 --loadLXMERT /local/harold/ubert/clip_vlp/lxmert/snap/pretrain/clip_full_20_no_qa_continue_from_17/Epoch11 --use_separate_optimizer_for_visual --sgd_lr 0.001 --sgd_momentum 0.0 --epoch 2 --schedule 1 --use_positional_embedding --pos_num 25

# bash run/finetune/snli_ve.bash 5 snap/snli/test 9595 1 --gradient_accumulation_steps 1 --batchSize 8 --lr 5e-5 --warmup_ratio 0.05 --report_step 400 --freeze_clip --use_positional_embedding --pos_num 25 