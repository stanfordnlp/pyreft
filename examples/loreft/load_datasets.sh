# clone the LLM-Adapters repository (we forked the version we used)
git clone --depth=1 https://github.com/aryamanarora/LLM-Adapters.git
# clone our own repository for holding ultrafeedback dataset
git clone --depth=1 https://github.com/frankaging/ultrafeedback-dataset.git

# move datasets
mv LLM-Adapters/dataset/ dataset/
mkdir dataset/commonsense_170k
mv LLM-Adapters/ft-training_set/commonsense_170k.json dataset/commonsense_170k/train.json
mkdir dataset/math_10k
mv LLM-Adapters/ft-training_set/math_10k.json dataset/math_10k/train.json
mkdir dataset/alpaca_data_cleaned
mv LLM-Adapters/ft-training_set/alpaca_data_cleaned.json dataset/alpaca_data_cleaned/train.json
mkdir dataset/ultrafeedback
mv ultrafeedback-dataset/train.json dataset/ultrafeedback/train.json

# clean
rm -rf LLM-Adapters
rm -rf ultrafeedback-dataset
