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
    unbuffer python -m torch.distributed.launch --master_port=$3 --nproc_per_node=$4 src/tasks/vqa.py \
    --distributed \
    --train train,nominival --valid minival  \
    --tqdm --output $output \
    --input_raw_images \
    --use_clip \
    --numWorkers 10 \
    --batchSize 32 --optim bert --lr 5e-5 --epochs 5 \
    --llayers 12 --xlayers 0 --rlayers 0 \
    --visualbert_style \
    --vqa_style_transform \
    --clip_model_name RN50x4 \
    --add_zero_padding \
    --gradient_accumulation_steps 8 \
    --loss_scale 500 \
    --warmup_ratio 0.05 \
    --report_step 400 \
    --use_separate_optimizer_for_visual \
    --sgd_lr 0.001 \
    --sgd_momentum 0.0 \
    --schedule 2 \
    --use_positional_embedding \
    --pos_num 25 \
    --fp16 \
    --clip_model_name RN50x4 \
    --loadLXMERTQA snap/pretrained/CLIP_VL_RN50x4
    ${@:5}  | tee $output/log.log


# CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
#     unbuffer python -m torch.distributed.launch --master_port=$3 --nproc_per_node=$4 src/tasks/vqa.py \
#     --distributed \
#     --train train,nominival --valid minival  \
#     --test test \
#     --tqdm --output $output \
#     --input_raw_images \
#     --use_clip \
#     --numWorkers 10 \
#     --batchSize 32 --optim bert --lr 5e-5 --epochs 5 \
#     --llayers 12 --xlayers 0 --rlayers 0 \
#     --visualbert_style \
#     --vqa_style_transform \
#     --clip_model_name RN50x4 \
#     --add_zero_padding \
#     --gradient_accumulation_steps 8 \
#     --loss_scale 500 \
#     --warmup_ratio 0.05 \
#     --report_step 400 \
#     --use_separate_optimizer_for_visual \
#     --sgd_lr 0.001 \
#     --sgd_momentum 0.0 \
#     --schedule 2 \
#     --use_positional_embedding \
#     --pos_num 25 \
#     --fp16 \
#     --clip_model_name RN50x4 \
#     --load snap/vqa/vqa_clip_rn50x4/BEST \
#     ${@:5}  | tee $output/log.log
    