# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

task=multitask_video

# or bart
model="bart"

echo $model

if [ $model == "t5" ]
then
    folder_prefix="VLT5"
    backbone="t5-base"
    batch_size=300
elif [ $model == "bart" ]
then
    folder_prefix="VLBart"
    backbone="facebook/bart-base"
    batch_size=50
fi

echo $folder_prefix
echo $backbone

feature=ViT

batch_size=40
#50
lr=2.4e-4
#3e-4

lora_dim=128

project_name=${feature}_LMsingle_dora${lora_dim}_bs${batch_size}_image224_video
run_name=dora_lora_setting_${lr}_${lora_dim}
output=snap/${folder_prefix}_${task}/$run_name

TOKENIZERS_PARALLELISM=True PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_port=26465 \
    src/${task}.py \
    --distributed --multiGPU \
    --optim adamw \
    --warmup_ratio 0.1 \
    --clip_grad_norm 5 \
    --lr ${lr} \
    --epochs 7 \
    --num_workers 4 \
    --backbone ${backbone} \
    --output $output ${@:2} \
    --num_beams 5 \
    --use_dora \
    --unfreeze_bias \
    --lora_settings \
    --lora_dim ${lora_dim} \
    --batch_size ${batch_size} \
    --valid_batch_size ${batch_size} \
    --use_tasks_prompts \
    --tasks "tvqa,how2qa,tvc,yc2c" \
    --feature ${feature} --n_boxes 64 --downsample \
    --image_size "(224,224)" \
    --project_name $project_name \
    --run_name $run_name

## this is for generating the output for submitting to https://value-benchmark.github.io/#:~:text=What%20is%20VALUE%3F,understanding%20both%20video%20and%20subtitles.
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_port=26465 \
    src/${task}.py \
    --distributed --multiGPU \
    --optim adamw \
    --warmup_ratio 0.1 \
    --clip_grad_norm 5 \
    --lr ${lr} \
    --epochs 0 \
    --num_workers 4 \
    --backbone ${backbone} \
    --output $output ${@:2} \
    --num_beams 5 \
    --use_dora \
    --load snap/${folder_prefix}_${task}/$run_name/LAST.pth \
    --unfreeze_bias \
    --lora_settings \
    --lora_dim ${lora_dim} \
    --batch_size ${batch_size} \
    --valid_batch_size ${batch_size} \
    --use_tasks_prompts \
    --tasks "tvqa,how2qa,tvc,yc2c" \
    --feature ${feature} --n_boxes 64 --downsample \
    --image_size "(224,224)" \
    --project_name $project_name \
    --run_name $run_name

