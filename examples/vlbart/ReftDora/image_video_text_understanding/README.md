# Finetuning VL-BART on image/video-text understaing tasks using DoRA

This directory includes the DoRA implementation and guidelines for reproducing the results in our paper.
We evaluate DoRA in a unified multi-task
setup on both image-text and video-text benchmarks following the settings of VL-Adapter. For the image-text tasks, we use four diverse V&L datasets: VQAv2, GQA, NLVR2, and MSCOCO image captioning. For video-text tasks, we use TVQA, How2QA, TVC, and YC2C. 

## Setup
```
# Create python environment
conda create -n vlt5 python=3.8
source activate vlt5

# Install python dependencies
pip install -r requirements.txt

# Download T5/BART backbone checkpoint
python download_backbones.py

# For MSCOCO captioning evaluation (optional; for captioning only)
python -c "import language_evaluation; language_evaluation.download('coco')"
```

## Data
```bash
# Store images, features, and annotations
./datasets
    COCO/
        images/
        clip_featuers/
    VG/
        images/
        clip_features/
    GQA/
        images/
        clip_features/
    nlvr/
        images/
        clip_features/
    vqa/
    lxmert/

    video/
        ann/
        vis_features

# Train VL-T5 with adapters
./VL-T5/
    src/
        multitask.py    <= multitask learning on 7 downstream tasks
        trainer_base.py <= DoRA implementation
```

### Image-text dataset
Please go to [link](https://drive.google.com/file/d/1O_RU1iFh_sbItZCTkOHUrbVIQQ_89Djj/view?usp=sharing) to download the processed CLIP features. We suggest to use [gdrive](https://github.com/prasmussen/gdrive) to download it. Unzip the downloaded file and arrange the folders according to the format demonstrated above.

If you would like to use dgrive to download the data, please try the following command

```
gdrive download 1O_RU1iFh_sbItZCTkOHUrbVIQQ_89Djj
```

### Extract your own CLIP features (Not necessary)
Please refer to `feature_extraction` for more details.

### Video-text dataset
Please go to [VALUE](https://github.com/VALUE-Leaderboard/DataRelease) to download the ViT processed data.

## Finetuning and Evaluation
### Finetuning VL-BART on Image-text datasets with DoRA (Evaluation included)
```
bash ./VL-T5/scripts/image/dora.sh 1
```
### Finetuning VL-BART on Video-text datasets with DoRA
```
bash ./VL-T5/scripts/video/dora.sh 1
```
### Evaluation of video-text tasks
Submit the generated test submission file strictly following the submission format (including directory layout and file names) specified [here](https://github.com/VALUE-Leaderboard/EvaluationTools) to the [Value benchmark website](https://value-benchmark.github.io/#:~:text=What%20is%20VALUE%3F,understanding%20both%20video%20and%20subtitles.) for evaluation.

## DoRA Result

### The multi-task evaluation results on VQA, GQA, NVLR2 and COCO Caption with the VL-BART backbone
| Method               |  # Params (%) | VQAv2 | GQA | NVLR2 | COCO Cap | Avg  |
|-----------------------|---------|--------|--------|-------------|--------------|---------|
| FT | 100 | 66.9 |56.7 | 73.7 |112.0| 77.3|
| LoRA | 5.93 |65.2 |53.6| 71.9| 115.3| 76.5|
| DoRA | 5.96 | 65.8 |54.7 |73.1 |115.9 | **77.4** |


### The multi-task evaluation results on TVQA, How2QA, TVC, and YC2C with the VL-BART backbone.
| Method                 |  # Params (%) |  TVQA | How2QA| TVC| YC2C | Avg  |
|-----------------------|---------|--------|--------|-------------|--------------|---------|
| FT | 100 | 76.3 | 73.9| 45.7| 154.0 | 87.5|
| LoRA | 5.17 | 75.5 | 72.9 | 44.6 | 140.9 | 83.5|
| DoRA | 5.19 |  76.3 | 74.1 | 45.8 | 145.4 | **85.4** |


## Acknowledgement
We greatly appreciate the contributions of [VL-Adapter](https://github.com/ylsung/VL_adapter) which has significantly benefited our work.