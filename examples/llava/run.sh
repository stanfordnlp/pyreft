# Experimenting. Please do NOT use.
python3 train_llava.py  \
    --output_dir=/nlp/scr/peterwz/outputs \
    --cache_dir=/nlp/scr/peterwz/.cache/ \
    --data_path=/nlp/scr/peterwz/datasets/data/llava_v1_5_mix665k.json \
    --image_folder=/nlp/scr/peterwz/datasets/data/ \
    --pretrain_mm_mlp_adapter=../../../llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin \
    --model_name_or_path=lmsys/vicuna-7b-v1.5