# Disruptive prediction model using KSTAR video and numerical data via Deep Learning
## Introduction
- This is git repository for research about predicting tokamak disruption prediction from video and 0D data using Deep Learning
- We used KSTAR IVIS dataset and 0D parameters from iKSTAR and finally we implemented the multi-modal model for (semi) real-time prediction
- Since video data have spatial-temporal information directly from the IVIS, the real-time prediction during the plasma operation can be possible without
any specific data preprocessing.
- We labeled the video data as a sequence of image data with different duration and distance (prediciton time) 
<div>
    <img src="/image/연구_소개_01.PNG"  width="640" height="196">
</div>

- We can proceed real-time disruption prediction using video data(left) and 0D data(right) for shot 21310. 
<div>
    <p float = 'left'>
        <img src="/results/real_time_disruption_prediction_21310.gif"  width="320" height="200">
        <img src="/results/real_time_disruption_prediction_0D_21310.gif"  width="320" height="200">
    </p>
</div>

- We also analyze the trained model by visualizing the latent vectors that the neural networks generate by compressing the data
- We can see that the prediction time is longer, the separation between disruptive and non-disruptive data decreases
<div>
    <p float = "left">
        <img src="/image/연구_소개_02.PNG"  width="640" height="224">
    </p>
</div>

- We also used attention rollout to visualize the attention matrix of the Video Vision Transformers to understand the importance of the video image to predict the disruption
- But, there would be no effective / important difference between two cases below. 

<div>
    <p float = 'left'>
        <img src="/image/연구_소개_03.PNG"  width="640" height="224">
    </p>
</div>

- We tried to show that video data would be helpful to detect VDE(Vertical Displacement Error) and time-varying shape characteristics.
- This means that we can effectivly predict the disruption with low false positive alarms with both video and 0D data since multi-modal learning is robust for data noise due to multi-modality

## How to Run
### setting
- Environment
```
conda create env -f environment.yaml
conda activate research-env
```

- Video Data Generation
```
# generate disruptive video data and normal video data from .avi
python3 ./src/generate_video_data.py --fps 210 --duration 21 --distance 5 --save_path './dataset/'

# train and test split with converting video as image sequences
python3 ./src/preprocessing.py --test_ratio 0.2 --valid_ratio 0.2 --video_data_path './dataset/dur21_dis0' --save_path './dataset/dur21_dis0'
```

- 0D Data(Numerical Data) Generation
```
# interpolate KSTAR data and convert as tabular dataframe
python3 ./src/generate_numerical_data.py 
```

### Training

- Models for video data
```
# ViViT model
python3 train_vivit.py --batch_size {batch size} --gpu_num {gpu num} --use_LDAM {bool : use LDAM loss}

# slowfast model
python3 train_slowfast.py --batch_size {batch size} --alpha {alpha} --gpu_num {gpu num} --use_LDAM {bool : use LDAM loss}

# R2Plus1D model
python3 train_R2Plus1D.py --batch_size {batch size} --gpu_num {gpu num} --use_LDAM {bool : use LDAM loss}
```

- Models for 0D data
```
# Conv-LSTM
python3 train_conv_lstm.py --batch_size {batch size} --gpu_num {gpu num} --use_LDAM {bool : use LDAM loss}

# Transformer
python3 train_ts_transformer.py --batch_size {batch size} --alpha {alpha} --gpu_num {gpu num} --use_LDAM {bool : use LDAM loss}
```

- Models for MultiModal(video + 0D data)
```
python3 train_multi_modal.py --batch_size {batch size} --alpha {alpha} --gpu_num {gpu num --use_LDAM {bool : use LDAM loss}
```

### Experiment
```
# experiment with different learning algorithm and models
python3 experiment.py --gpu_num {gpu_num} --loss_type {'CE', 'FOCAL', 'LDAM'}
```

## Detail
### Model to use
- Video encoder
    - R2Plus1D
    - Slowfast
    - ViViT (selected)

- 0D data encoder
    - Transformer
    - Conv1D-LSTM using self-attention (selected)

- Multimodal Model
    - Multimodal fusion model: video encoder + 0D data encoder
    - Tensor Fusion Network
    - Other methods (Future work)
        - Multimodal deep representation learning for video classification : https://link.springer.com/content/pdf/10.1007/s11280-018-0548-3.pdf?pdf=button
        - Truly Multi-modal YouTube-8M Video Classification with Video, Audio, and Text : https://static.googleusercontent.com/media/research.google.com/ko//youtube8m/workshop2017/c06.pdf

### Technique or algorithm to use
- Solving imbalanced classificatio issue
    - Re-Sampling : ImbalancedWeightedSampler, Over-Sampling for minor classes
    - Re-Weighting : Define inverse class frequencies as weights to apply with loss function (CE, Focal Loss, LDAM Loss)
    - LDAM with DRW : Label-distribution-aware margin loss with deferred re-weighting scheduling
    - Multimodal Learning : Gradient Blending for avoiding sub-optimal due to large modalities
    - Multimodal Learning : CCA Learning for enhancement

- Analysis on physical characteristics of disruptive video data
    - CAM
    - Grad CAM
    - attention rollout (selected)

- Data augmentation
    - Video Mixup Algorithm for Data augmentation(done, not effective)
    - Conventional Image Augmentation(Flip, Brightness, Contrast, Blur, shift)

- Training Process enhancement
    - Multigrid training algorithm : Fast training for SlowFast
    - Deep CCA : Deep cannonical correlation analysis to train multi-modal representation

- Generalization and Robustness
    - Add noise with image sequence and 0D data for robustness
    - Multimodality can also guarantee the robustness from noise of the data
    - Gradient Blending for avoiding sub-optimal states from multi-modal learning

### Additional Task
- Multi-GPU distributed Learning : done
- Database contruction : Tabular dataset(IKSTAR) + Video dataset, done
- ML Pipeline : Tensorboard (not yet)

### Dataset
- Disruption : disruptive state at t = tipminf (current-quench)
- Borderline : inter-plane region (not used)
- Normal : non-disruptive state

## Reference
- R2Plus1D : A Spatial-temporal Attention Module for 3D Convolution Network in Action Recognition
- Slowfast : SlowFast Networks for Video Recognition
- Video Vision Transformer : ViViT: A Video Vision Transformer, Anurag Arnab et al, 2021
- Multigrid : A Multigrid Method for Efficiently Training Video Models, Chao-Yuan Wu et al, 2020
- Video Data Augmentation : VideoMix: Rethinking Data Augmentation for Video Classification
- LDAM : Label-distribution-aware Margin Loss
- Focal Loss : Focal Loss for Dense object detection, TY Lin et al, 2017
- Gradient Blending : What Makes Training Multi-Modal Classification Networks Hard?, Weiyao Wang et al, 2022
- Tensor Fusion Network : Tensor Fusion Network for Multimodal Sentiment Analysis, Amir Zadeh et al, 2017