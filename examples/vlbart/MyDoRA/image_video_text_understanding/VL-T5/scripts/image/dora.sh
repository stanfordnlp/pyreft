# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

export CUDA_VISIBLE_DEVICES=0
task=multitask

# or bart
model="bart"

echo $model

if [ $model == "t5" ]
then
    folder_prefix="VLT5"
    backbone="t5-base"
    batch_size=400 # 400
elif [ $model == "bart" ]
then
    folder_prefix="VLBart"
    backbone="facebook/bart-base"
    batch_size=300 # 300
fi

echo $folder_prefix
echo $backbone

feature=RN101

lr=1e-3

lora_dim=128

project_name=${feature}_LMsingle_dora_${lora_dim}_bs${batch_size}_image224_lora_settings
run_name=tune+lr${lr}_plzplz2
output=snap/${folder_prefix}_${task}/$run_name

TOKENIZERS_PARALLELISM=True PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_port=26468 \
    src/${task}.py \
    --distributed --multiGPU \
    --optim adamw \
    --warmup_ratio 0.00 \
    --weight_decay 0.005 \
    --lr ${lr} \
    --epochs 20 \
    --num_workers 4 \
    --backbone ${backbone} \
    --output $output ${@:2} \
    --num_beams 5 \
    --use_tasks_prompts \
    --batch_size ${batch_size} \
    --valid_batch_size ${batch_size} \
    --tasks "vqa" \
    --dropout 0.00 \
    --reft_rank 4 \
    --unfreeze_bias \
    --unfreeze_layer_norms \
    --positions "f11+l11" \
    --feature ${feature} --n_boxes 36 --downsample \
    --image_size "(224,224)" \
    --project_name $project_name \
    --run_name $run_name
