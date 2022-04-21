# Plasma Disruption Predictive model using Plasma Image dataset
## Introduction

<img src="/image/연구_소개_01.PNG"  width="900" height="224">
<p>2차원 플라즈마 이미지 분석을 통한 토카막 플라즈마 붕괴 실시간 예측 연구</p>

## How to Run(Test)
```
conda create env -f environment.yaml
conda activate research-env
python3 {filename.py} # (ex : python3 train_slowfast.py)
```
## Code Structure
```
```

## Detail
### model for use
1. SITS-BERT 
2. R2Plus1D
3. Slowfast
4. UTAE
5. R3D
### algorithm
1. Multigrid training algorithm(proceeding)
2. Multi-GPU distributed Learning(done)
3. Video Mixup Algorithm for Data augmentation(done)
### Dataset
1. Disruption : disrupted after tfQ-end reached
2. Borderline : inter-plane region 
3. Normal : not disrupted 
## Reference
- R2Plus1D : A Spatial-temporal Attention Module for 3D Convolution Network in Action Recognition
- Slowfast : SlowFast Networks for Video Recognition
- Multigrid : A Multigrid Method for Efficiently Training Video Models, Chao-Yuan Wu et al, 2020
- Video Data Augmentation : VideoMix: Rethinking Data Augmentation for Video Classification
- SITS-BERT : Self-Supervised pretraining of Transformers for Satellite Image Time Series Classification
- UTAE : Panoptic Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks