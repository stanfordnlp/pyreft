# clone the LLM-Adapters repository (we forked the version we used)
git clone --depth=1 https://github.com/aryamanarora/LLM-Adapters.git

# move datasets
mv LLM-Adapters/dataset/ dataset/
mkdir dataset/commonsense_170k
mv LLM-Adapters/ft-training_set/commonsense_170k.json dataset/commonsense_170k/train.json
mkdir dataset/math_10k
mv LLM-Adapters/ft-training_set/math_10k.json dataset/math_10k/train.json

# clean
rm -rf LLM-Adapters