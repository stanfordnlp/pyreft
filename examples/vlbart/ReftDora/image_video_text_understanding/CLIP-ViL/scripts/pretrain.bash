# The name of this experiment.
name=$2

# Save logs and models under snap/vqa; make backup.
output=snap/pretrain/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    unbuffer python -m torch.distributed.launch --master_port=$3 --nproc_per_node=$4 src/pretrain/lxmert_pretrain.py \
    --taskMaskLM --taskMatched \
    --visualLosses obj,attr,feat \
    --wordMaskRate 0.15 \
    --train mscoco_train,mscoco_nominival,vgnococo --valid mscoco_minival \
    --batchSize 256 --optim bert --lr 1e-4 --epochs 20 \
    --tqdm \
    --llayers 12 --xlayers 0 --rlayers 0 \
    --visualbert_style \
    --input_raw_images \
    --vqa_style_transform \
    --objMaskRate 0.0 \
    --numWorkers 0\
    --clip_model_name RN50\
    --use_clip \
    --distributed \
    --output $output\
    ${@:5}  | tee $output/log.log
