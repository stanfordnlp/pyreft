git clone --depth=1 https://github.com/aryamanarora/LLM-Adapters.git
git clone --depth=1 https://github.com/frankaging/ultrafeedback-dataset.git

move LLM-Adapters\dataset dataset
mkdir dataset\commonsense_170k
move LLM-Adapters\ft-training_set\commonsense_170k.json dataset\commonsense_170k\train.json
mkdir dataset\math_10k
move LLM-Adapters\ft-training_set\math_10k.json dataset\math_10k\train.json
mkdir dataset\ultrafeedback
move ultrafeedback-dataset\train.json dataset\ultrafeedback\train.json

rd /s /q LLM-Adapters
rd /s /q ultrafeedback-dataset
