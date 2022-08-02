# Disruptive prediction model using KSTAR video and numerical data via Deep Learning
## Introduction
<img src="/image/연구_소개_01.PNG"  width="900" height="224">
<p>Research for predicting tokamak plasma disruption from video and numerical data via Deep Learning</p>

## How to Run
### setting
```
conda create env -f environment.yaml
conda activate research-env
python3 {filename.py} # (ex : python3 train_slowfast.py)
```

### training
```
python3 train_model.py --batch_size {batch size} --gpu_num {gpu num} --model_name {model name} --use_LDAM {bool : use LDAM loss} --use_mixup {bool : use video mixup algorithm}
```

### test
```
python3 test_model.py --video_file {video file path} --model_name {model name} --model_weight {path for loading model weights}
```

## Detail
### model to use
1. SITS-BERT 
2. R2Plus1D
3. Slowfast
4. UTAE
5. R3D
6. VAT

### technique or algorithm to use
1. Solving imbalanced classificatio issue
- Adversarial Training 
- LDAM with DRW : Label-distribution-aware margin loss with deferred re-weighting scheduling

2. Analysis on physical characteristics of disruptive video data
- CAM
- Grad CAM

3. Data augmentation
- Video Mixup Algorithm for Data augmentation(done, not effective)

4. Training Process enhancement
- Multigrid training algorithm

### Additional Task
- Multi-GPU distributed Learning : done
- Database contruction : Tabular dataset(IKSTAR) + Video dataset, not yet
- ML Pipeline : not yet

### Dataset
1. Disruption : disrupted after tfQ-end reached
2. Borderline : inter-plane region 
3. Normal : not disrupted 

### Code Structure
```
```

## Reference
- R2Plus1D : A Spatial-temporal Attention Module for 3D Convolution Network in Action Recognition
- Slowfast : SlowFast Networks for Video Recognition
- Multigrid : A Multigrid Method for Efficiently Training Video Models, Chao-Yuan Wu et al, 2020
- Video Data Augmentation : VideoMix: Rethinking Data Augmentation for Video Classification
- SITS-BERT : Self-Supervised pretraining of Transformers for Satellite Image Time Series Classification
- UTAE : Panoptic Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks