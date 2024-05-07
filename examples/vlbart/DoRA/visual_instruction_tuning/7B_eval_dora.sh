# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

## vqav2
CKPT=$1
CKPT_PATH=$2
EVALPATH=$3
RESULTPATH=$4
BASEPATH=$5
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}
SPLIT="llava_vqav2_mscoco_test-dev2015"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $CKPT_PATH \
        --model-base lmsys/vicuna-7b-v1.5 \
        --question-file $EVALPATH/vqav2/$SPLIT.jsonl \
        --image-folder $EVALPATH/vqav2/test2015 \
        --answers-file $EVALPATH/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

mkdir -p $RESULTPATH/vqav2/answers/$SPLIT/$CKPT/
output_file=$RESULTPATH/vqav2/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"


# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $EVALPATH/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT --save_dir $RESULTPATH/vqav2 --eval_dir $EVALPATH/vqav2

## gqa
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
SPLIT="llava_gqa_testdev_balanced"
GQADIR="$EVALPATH/gqa/data"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $CKPT_PATH \
        --model-base lmsys/vicuna-7b-v1.5 \
        --question-file $EVALPATH/gqa/$SPLIT.jsonl \
        --image-folder $EVALPATH/gqa/data/images \
        --answers-file $EVALPATH/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=$EVALPATH/gqa/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $EVALPATH/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# mkdir -p $GQADIR/

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
mkdir -p $RESULTPATH/$CKPT
python $GQADIR/eval/eval.py --tier testdev_balanced|tee -a $RESULTPATH/$CKPT/gqa_eval.txt

cd $BASEPATH
## viswiz
python -m llava.eval.model_vqa_loader \
    --model-path $CKPT_PATH \
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file $EVALPATH/vizwiz/llava_test.jsonl \
    --image-folder $EVALPATH/vizwiz/test \
    --answers-file $EVALPATH/vizwiz/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p $RESULTPATH/vizwiz/answers_upload

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file $EVALPATH/vizwiz/llava_test.jsonl \
    --result-file $EVALPATH/vizwiz/answers/$CKPT.jsonl \
    --result-upload-file $RESULTPATH/vizwiz/answers_upload/$CKPT.json

## sqa
python -m llava.eval.model_vqa_science \
    --model-path $CKPT_PATH \
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file $EVALPATH/scienceqa/llava_test_CQM-A.json \
    --image-folder $EVALPATH/scienceqa/images/test \
    --answers-file $EVALPATH/scienceqa/answers/$CKPT.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p $RESULTPATH/scienceqa/$CKPT/

python llava/eval/eval_science_qa.py \
    --base-dir $EVALPATH/scienceqa \
    --result-file $EVALPATH/scienceqa/answers/$CKPT.jsonl \
    --output-file $EVALPATH/scienceqa/answers/$CKPT-output.jsonl \
    --output-result $EVALPATH/scienceqa/answers/$CKPT-result.json|tee -a $RESULTPATH/scienceqa/$CKPT/sqa_eval.txt

## t-vqa
python -m llava.eval.model_vqa_loader \
    --model-path $CKPT_PATH \
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file $EVALPATH/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder $EVALPATH/textvqa/train_images \
    --answers-file $EVALPATH/textvqa/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p $RESULTPATH/textvqa/$CKPT/

python -m llava.eval.eval_textvqa \
    --annotation-file $EVALPATH/textvqa/TextVQA_0.5.1_val.json \
    --result-file $EVALPATH/textvqa/answers/$CKPT.jsonl|tee -a $RESULTPATH/textvqa/$CKPT/textvqa_eval.txt

# pope
python -m llava.eval.model_vqa_loader \
    --model-path $CKPT_PATH \
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file $EVALPATH/pope/llava_pope_test.jsonl \
    --image-folder $EVALPATH/pope/val2014 \
    --answers-file $EVALPATH/pope/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p $RESULTPATH/pope/$CKPT

python llava/eval/eval_pope.py \
    --annotation-dir $EVALPATH/pope/coco \
    --question-file $EVALPATH/pope/llava_pope_test.jsonl \
    --result-file $EVALPATH/pope/answers/$CKPT.jsonl|tee -a $RESULTPATH/pope/$CKPT/pope_eval.txt

## mm-bench
SPLIT="mmbench_dev_20230712"

python -m llava.eval.model_vqa_mmbench \
    --model-path $CKPT_PATH \
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file $EVALPATH/mmbench/$SPLIT.tsv \
    --answers-file $EVALPATH/mmbench/answers/$SPLIT/$CKPT.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p $RESULTPATH/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file $EVALPATH/mmbench/$SPLIT.tsv \
    --result-dir $EVALPATH/mmbench/answers/$SPLIT \
    --upload-dir $RESULTPATH/mmbench/answers_upload/$SPLIT \
    --experiment $CKPT

