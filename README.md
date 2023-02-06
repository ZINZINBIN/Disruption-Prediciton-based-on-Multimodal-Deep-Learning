# Disruptive prediction model using KSTAR video and numerical data via Deep Learning
## Introduction
- This is git repository for research about predicting tokamak disruption prediction from video and 0D data using Deep Learning
- We used KSTAR IVIS dataset and 0D parameters from iKSTAR and finally we implemented the multi-modal model for (semi) continuous prediction
- Since video data have spatial-temporal information directly from the IVIS, the real-time prediction during the plasma operation can be possible without any specific data preprocessing.
- We labeled the video data as a sequence of image data with different duration and distance (prediciton time) 

### How to generate training data
<div>
    <img src="/image/연구_소개_01.PNG"  width="640" height="196">
</div>

- We set image sequence data as input tensor and label the fadeout frame as disruptive phase 
- We set the last second frame of the video as current quench and predict the disruption before the current quench happens
- This data generation method can be useful for small dataset since it uses all data obtained from each video
- However, the severe imbalance data distribution due to the time scale of disruptive phase can happen
- Therefore, we use specific learning algorithms to handle this issue

### The model performance of disruption prediction
<div>
    <p float = 'left'>
        <img src="/results/real_time_disruption_prediction_21310.gif"  width="320" height="200">
        <img src="/results/real_time_disruption_prediction_0D_21310.gif"  width="320" height="200">
    </p>
</div>

- We can proceed real-time disruption prediction using video data(left) and 0D data(right) for shot 21310. 
- The sensitivity of the model controled by the threshold affects the missing alarm rate
- Each data has different characteristics for disruption prediction : How about combining two data at once?

### Analysis of the models using visualization of embedding space
<div>
    <p float = "left">
        <img src="/image/연구_소개_02.PNG"  width="640" height="224">
    </p>
</div>

- We also analyze the trained model by visualizing the latent vectors that the neural networks generate by compressing the data
- We can see that the prediction time is longer, the separation between disruptive and non-disruptive data decreases

### Analysis of the models using attention rollout for vision model
<div>
    <p float = 'left'>
        <img src="/image/연구_소개_03.PNG"  width="640" height="224">
    </p>
</div>

- We also used attention rollout to visualize the attention matrix of the Video Vision Transformers to understand the importance of the video image to predict the disruption
- But, there would be no effective / important difference between two cases below. 

### Summary
- We tried to show that video data would be helpful to detect VDE(Vertical Displacement Error) and time-varying shape characteristics.
- This means that we can effectivly predict the disruption with low false positive alarms with both video and 0D data since multi-modal learning is robust for data noise due to multi-modality
- The result of multi-modal data will soon be showned layer

## How to Run
### setting
- Environment
    ```
    conda create env -f environment.yaml
    conda activate research-env
    ```

- Video Dataset Generation : old version, inefficient memory usage and scalability
    ```
    # generate disruptive video data and normal video data from .avi
    python3 ./src/generate_video_data.py    --fps 210 
                                            --duration 21 
                                            --distance 5 
                                            --save_path './dataset/'

    # train and test split with converting video as image sequences
    python3 ./src/preprocessing.py  --test_ratio 0.2 
                                    --valid_ratio 0.2 
                                    --video_data_path './dataset/dur21_dis0' 
                                    --save_path './dataset/dur21_dis0'
    ```

- Video Dataset Generation : new version, more efficient than old version
    ```
    # additional KSTAR shot log with frame information of the video data
    python3 ./src/generate_modified_shot_log.py

    # generate video dataset from extended KSTAR shot log : you don't need to split the train-test set for every distance
    python3 ./src/generate_video_data2.py   --fps 210
                                            --raw_video_path "./dataset/raw_videos/raw_videos/"
                                            --df_shot_list_path "./dataset/KSTAR_Disruption_Shot_List_extend.csv"
                                            --save_path "./dataset/temp"
                                            --width 256
                                            --height 256
                                            --overwrite True
    ```

- 0D Dataset Generation (Numerical dataset)
    ```
    # interpolate KSTAR data and convert as tabular dataframe
    python3 ./src/generate_numerical_data.py 
    ```

### Test
- Test code before model training : check the invalid data or issues from model architecture
    ```
    # test all process : data + model
    pytest test

    # test the data validity
    pytest test/test_data.py

    # test the model validity
    pytest test/test_model.py
    ```

### Model training process
- Models for video data
    ```
    python3 train_vision_nework.py --batch_size {batch size} --gpu_num {gpu num} 
                                    --use_LDAM {bool : use LDAM loss} --model_type {model name} 
                                    --tag {name of experiment / info} --use_DRW {bool : use Deferred re-weighting} 
                                    --use_RS {bool : use re-sampling} --seq_len {int : input sequence length} 
                                    --pred_len {int : prediction time} --image_size {int}
    ```

- Models for 0D data
    ```
    python3 train_0D_nework.py --batch_size {batch size} --gpu_num {gpu num} 
                                --use_LDAM {bool : use LDAM loss} --model_type {model name} 
                                --tag {name of experiment / info} --use_DRW {bool : use Deferred re-weighting} 
                                --use_RS {bool : use re-sampling} --seq_len {int : input sequence length} 
                                --pred_len {int : prediction time}
    ```

- Models for MultiModal(video + 0D data)
    ```
    python3 train_multi_modal.py --batch_size {batch size} --gpu_num {gpu num} 
                                --use_LDAM {bool : use LDAM loss} --use_GB {bool : use Deferred re-weighting} 
                                --tag {name of experiment / info} --use_DRW {bool : use Deferred re-weighting} 
                                --use_RS {bool : use re-sampling} --seq_len {int : input sequence length} 
                                --pred_len {int : prediction time}
    ```

### Experiment
- Experiment for each network(vision, 0D, multimodal) with different prediction time
    ```
    # R1Plus1D
    bashrc ./exp_r1plus1d.sh

    # Slowfast
    bashrc ./exp_slowfast.sh

    # ViViT
    bashrc ./exp_vivit.sh

    # Transformer
    bashrc ./exp_0D.sh

    # Multimodal model
    bashrc ./exp_multi.sh

    # Multimodal model with Gradient Blending
    bashrc ./exp_multi_gb.sh
    ```

- Experiment with different learning algorithms and models
    ```
    # use python file
    python3 experiment.py --gpu_num {gpu_num} --loss_type {'CE', 'FOCAL', 'LDAM'}

    # use bash file
    bashrc ./exp_learning_algorithm.sh
    ```

## Detail
### Model to use
- Video encoder
    - R2Plus1D : https://github.com/irhum/R2Plus1D-PyTorch
    - Slowfast : https://github.com/facebookresearch/SlowFast 
    - ViViT : https://github.com/rishikksh20/ViViT-pytorch

- 0D data encoder
    - Transformer : paper(https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf), application code(https://www.kaggle.com/general/200913)
    - Conv1D-LSTM using self-attention : https://pseudo-lab.github.io/Tutorial-Book/chapters/time-series/Ch5-CNN-LSTM.html
    - MLSTM_FCN : paper(https://arxiv.org/abs/1801.04503), application code(https://github.com/titu1994/MLSTM-FCN)

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
    - CAM : proceeding
    - Grad CAM : paper(https://arxiv.org/abs/1610.02391), target model(R2Plus1D, SlowFast)
    - attention rollout : paper(https://arxiv.org/abs/2005.00928), target model(ViViT)

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
- ML Pipeline : Tensorboard, done

### Dataset
- Disruption : disruptive state at t = tipminf (current-quench)
- Borderline : inter-plane region (not used)
- Normal : non-disruptive state

## Reference
- R2Plus1D : A Spatial-temporal Attention Module for 3D Convolution Network in Action Recognition(https://arxiv.org/abs/1711.11248)
- Slowfast : SlowFast Networks for Video Recognition, Christoph Feichtenhofer et al, 2018(https://arxiv.org/abs/1812.03982)
- Video Vision Transformer : ViViT: A Video Vision Transformer, Anurag Arnab et al, 2021(https://arxiv.org/pdf/2103.15691.pdf)
- Multigrid : A Multigrid Method for Efficiently Training Video Models, Chao-Yuan Wu et al, 2020
- Video Data Augmentation : VideoMix: Rethinking Data Augmentation for Video Classification
- LDAM : Label-distribution-aware Margin Loss
- Focal Loss : Focal Loss for Dense object detection, TY Lin et al, 2017
- Gradient Blending : What Makes Training Multi-Modal Classification Networks Hard?, Weiyao Wang et al, 2022
- Tensor Fusion Network : Tensor Fusion Network for Multimodal Sentiment Analysis, Amir Zadeh et al, 2017