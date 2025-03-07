 # Disruptive prediction model using KSTAR video and numerical data via Deep Learning
<a href = "https://www.sciencedirect.com/science/article/pii/S0920379624000577" target = "_blank">[Paper : Disruption Prediction and Analysis Through Multimodal Deep Learning in KSTAR]</a>

## Introduction
<div>
    This is github repository of research on disruption predictin using deep learning in KSTAR dataset. In this research, KSTAR IVIS data and 0D parameters are used for predicting disruptions. We obtained plasma image data from IVIS in KSTAR and 0D parameters such as stored energy, beta, plasma current, internal inductance and so on. Additonal information such as ECE data including electron temperature and density are also used. 
</div>
<div>
    <img src="/image/연구_소개_컨셉.PNG"  width="640" height="320">
</div>
<div>
    Unlike other research for disruption predictin using machine learning, we also used video data as an input so to use spatial-temporal information of plasma including time-varying plasma shape and light emission induced by plasma-neutral interaction. This requires neural networks which are generally used for video classification task. 
</div>
<div>
    <p float = 'left'>
        <img src="/image/연구_소개_08.PNG"  width="320" height="200">
        <img src="/image/연구_소개_11.PNG"  width="320" height="200">
    </p>
</div>
<div>
    However, there is imbalance data distribution issue which results from the time scale difference between disruptive phase and plasma operation. Thus, we applied resampling with Focal loss and LDAM loss to handle this problem. We demonstrated that using multimodal data including video and 0D parameters can enhance the precision of the disruption alarms. Moreover, some consistent results can be shown using GradCAM and permutation feature importance, which implies that the networks focus on the near of the core plasma from both image and 0D parameters(Te, Ne in the core of plasma). Several techniques were used for comparing the model performance indirectly. 
</div>
<div>
    <img src="/image/연구_소개_00.PNG"  width="640" height="256">
</div>
<div>
    If there is any question or comment, please contact my email (personal : wlstn5376@gmail.com, school : asdwlstn@snu.ac.kr) whenever you want.
</div>
<p></p>

### How to generate training data
<div>
    <img src="/image/연구_소개_01.PNG"  width="640" height="196">
</div>
<div>To generate training dataset, we have to use some assumptions and technique. The major process can be listed as below.</div>

- Firstly, we set image sequence data as an input data (B,T,C,W,H) and assumed that the last frame where the image of the plasma in tokamak disapper is a disruption. 
- Then, the last second frame of the image sequence can be considered as a current quench.
- Thus, the frame sequences including the last second frame of each experiment data, are labeled as disruptive. 
- Under this condition, the neural networks trained by these labeled dataset can predict the disruption prior to a current quench.

<div>
    This method can be useful for small dataset since it uses all data obtained from each experiment, but imbalance data distribution due to the time scale of disruptive phase can occur in this process. Therefore, we use specific learning algorithms to handle this issue (e.g. Re-sampling, Re-weighting, Focal Loss, LDAM Loss)
</div>
<p></p>

### The model performance of disruption prediction
<div>
    <p float = 'left'>
        <img src="/results/real_time_disruption_prediction_21310.gif"  width="320" height="200">
        <img src="/results/real_time_disruption_prediction_0D_21310.gif"  width="320" height="200">
    </p>
</div>
<div>
    We can proceed continuous disruption prediction using video data(left) and 0D data(right) for KSTAR shot #21310. It is quite obvious that the sensitivity of the model controled by the threshold affects the missing alarm rate(recall). Additionally, different characteristics in predicting disruption are observed according to the data modality. Thus, we can think about combining two different modality of data so to overcome the limits. 
</div>
<p></p>

### Analysis of the models using visualization of hidden vectors
<div>
    <p float = "left">
        <img src="/image/연구_소개_02.PNG"  width="640" height="320">
    </p>
</div>
<div>
    The hidden vectors embedded by vision networks can be visualized using PCA or t-SNE. The separation of the embedded data would seem to be more clear as the model predicts the disruption as well. Since the distinct patterns or precurssors for predicting disruptions can not be detected over long prediction time, the separtion is hardly observed at the case of long prediction time.
</div>
<p></p>

### Analysis of the models using permutation feature importance
<div>
    <p float = "left">
        <img src="/image/연구_소개_12.PNG"  width="640" height="300">
    </p>
</div>
<div>
    The importance of the 0D parameters can be estimated by permutation feature importance. According to the permuatation feature importance, we can observe that the electron information such as electron density and temperatature from both edge and core is important to predict the disruption. It is quite interesting that the importance of q95 is smaller than other values except kappa. Since KSTAR operations proceed in high q95 region, low q-limit is not considerable than other factors.  
</div>
<p></p>

### Analysis of the models using GradCAM and attention rollout
<div>
    <p float = 'left'>
        <img src="/image/연구_소개_03.PNG"  width="640" height="280">
    </p>
</div>
<div>
    GradCAM and attention rollout are applied to visualize the information flow in case of models trained by IVIS data. CNN-based networks are highly trained as well so to focus on the specific region of the inside of the tokamak. However, this locality can not be observed in case of Video Vision Transformer, which has low precision due to high false alarm rate.  
</div>
<p></p>

### Enhancement of predicting disruptions using multimodal learning
<div>
    <p float = 'left'>
        <img src="/image/연구_소개_09.PNG"  width="360" height="240">
        <img src="/image/연구_소개_10.PNG"  width="360" height="240">
    </p>
</div>
<div>
    Applying multimodal learning to the video vision network shows decrease on false alarms and locality in the vision encoder. This means that the modal capabilities increased by multimodal data can enhance the low precision of the disruption predictors and show robustness with respect to the data noise. Furthermore, several factors which induce the various types of disruptions can be considered by using multimodal data including IVIS, 1D profiles, 0D parameters and so on.
</div>
<p></p>

## Enviornment
<p>The code was developed using python 3.9 on Ubuntu 18.04</p>
<p>The GPU used : NVIDIA GeForce RTX 3090 24GB x 4</p>
<p>The resources for training networks were provided by <a href = "http://fusma.snu.ac.kr/plare/" target = "_blank">PLARE</a> in Seoul National University</p>

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
    python3 ./src/generate_video_data_fixed.py  --fps 210
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
    python3 ./src/generate_video_data.py    --fps 210
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
                                --pred_len {int : prediction time} --tau {int : stride for input sequence}
    ```

### Experiment
- Experiment for each network(vision, 0D, multimodal) with different prediction time
    ```
    # R1Plus1D
    sh exp/exp_r1plus1d.sh

    # Slowfast
    sh exp/exp_slowfast.sh

    # ViViT
    sh exp/exp_vivit.sh

    # Transformer
    sh exp/exp_0D_transformer.sh

    # CnnLSTM
    sh exp/exp_0D_cnnlstm.sh

    # MLSTM-FCN
    sh exp/exp_0D_mlstm.sh

    # Multimodal model
    sh exp/exp_multi.sh

    # Multimodal model with Gradient Blending
    sh exp/exp_multi_gb.sh
    ```

- Experiment with different learning algorithms and models
    ```
    # case : R2Plus1D
    sh exp/exp_la_r2plus1d.sh

    # case : SlowFast
    sh exp/exp_la_slowfast.sh

    # case : ViViT
    sh exp/exp_la_vivit.sh
    ```

- Model performance visualization for continuous disruption prediction using gif
    ```
    python3 make_continuous_prediction.py
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

## 📖 Citation
If you use this repository in your research, please cite the following:

### 📜 Research Article
[Disruption prediction and analysis through multimodal deep learning in KSTAR](https://doi.org/10.1016/j.fusengdes.2024.114204)  
Kim, Jinsu, et al. "Disruption prediction and analysis through multimodal deep learning in KSTAR." Fusion Engineering and Design 200 (2024): 114204.

### 📌 Code Repository
Jinsu Kim (2024). **Disruption-Prediciton-based-on-Multimodal-Deep-Learning**. GitHub.  
[https://github.com/ZINZINBIN/Disruption-Prediciton-based-on-Multimodal-Deep-Learning](https://github.com/ZINZINBIN/Disruption-Prediciton-based-on-Multimodal-Deep-Learning)

#### 📚 BibTeX:
```bibtex
@software{Kim_Bayesian_Deep_Learning_2024,
author = {Kim, Jinsu},
doi = {https://doi.org/10.1088/1361-6587/ad48b7},
license = {MIT},
month = may,
title = {{Bayesian Deep Learning based Disruption Prediction Model}},
url = {https://github.com/ZINZINBIN/Bayesian-Disruption-Prediction},
version = {1.0.0},
year = {2024}
}
```