''' Heatmap for R2Plus1D and SlowFast model
    GradCAM will be used : Due to the fact that cam has limit on use, we will use only GradCAM
    
    Reference
    - Simple description : https://tyami.github.io/deep%20learning/CNN-visualization-Grad-CAM/
    - Review of the GradCAM paper : https://cumulu-s.tistory.com/40
    - source code : https://github.com/jacobgil/pytorch-grad-cam
    - source code 2 : https://cumulu-s.tistory.com/41
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

# https://cumulu-s.tistory.com/41 -> Good source!
class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(self, input_img:torch.Tensor):
        # input image 기준으로 양수인 부분만 1로 만드는 positive_mask 생성
        positive_mask = (input_img > 0).type_as(input_img)
        
        # torch.addcmul(input, tensor1, tensor2) => output = input + tensor1 x tensor 2
        # input image와 동일한 사이즈의 torch.zeros를 만든 뒤, input image와 positive_mask를 곱해서 output 생성
        output = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), input_img, positive_mask)
        
        # backward에서 사용될 forward의 input이나 output을 저장
        self.save_for_backward(input_img, output)
        return output

    @staticmethod
    def backward(self, grad_output:torch.Tensor):
        
        # forward에서 저장된 saved tensor를 불러오기
        input_img, output = self.saved_tensors
        grad_input = None

        # input image 기준으로 양수인 부분만 1로 만드는 positive_mask 생성
        positive_mask_1 = (input_img > 0).type_as(grad_output)
        
        # 모델의 결과가 양수인 부분만 1로 만드는 positive_mask 생성
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        
        # 먼저 모델의 결과와 positive_mask_1과 곱해주고,
        # 다음으로는 positive_mask_2와 곱해줘서 
        # 모델의 결과가 양수이면서 input image가 양수인 부분만 남도록 만들어줌
        grad_input = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img),
                                   torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), grad_output,
                                                 positive_mask_1), positive_mask_2)
        return grad_input
    
# GradCAM Wrapper for R2Plus1D
class GradCAM_R2Plus1D:
    def __init__(self, model : nn.Module):
        super().__init__()
        self.model = model
        
        # activations
        self.feature_blobs = []
        
        # gradients
        self.backward_featuer = []
        
        def _hook_feature(module:nn.Module, input:torch.Tensor, output:torch.Tensor):
            self.feature_blobs.append(output.cpu().data)
            
        # Grad-CAM
        def _backward_hook(module:nn.Module, input:torch.Tensor, output:torch.Tensor):
            self.backward_feature.append(output[0])

        self.model._modules['res2plus1d'].get_submodule("conv5").register_forward_hook(_hook_feature)
        self.model._modules['res2plus1d'].get_submodule("conv5").register_backward_hook(_backward_hook)
        
        self.model.eval()

    def __call__(self, video : torch.Tensor, title : Optional[str] = None, save_dir :Optional[str] = None):
        logit = self.model(video)
        output = torch.nn.functional.softmax(logit, dim = 1).data.squeeze()
        
        score = logit[:,0].squeeze()
        score.backward(retain_graph = True)
        
        activations = self.feature_blobs[-1]
        gradients = self.backward_feature[-1]
        
        alpha = gradients.mean(2) # time axis average
        alpha = alpha.view(1, 128, -1).mean(2) # average for all spatial axis
        weights = alpha.view(1, 128, 1, 1, 1)

        grad_cam_map = (weights*activations).sum(1, keepdim = True)
        grad_cam_map = F.relu(grad_cam_map).view(1,1,3,8,8)

        grad_cam_map_interp = torch.empty((1,1,3,128,128))

        for idx in range(3):
            grad_cam_map_interp[:,:,idx,:,:] = F.interpolate(grad_cam_map[:,:,idx,:,:], size=(128, 128), mode='bilinear', align_corners=False)

        grad_cam_map_interp = grad_cam_map_interp.mean(2)
        map_min, map_max = grad_cam_map_interp.min(), grad_cam_map_interp.max()
        grad_cam_map = (grad_cam_map_interp - map_min).div(map_max - map_min).data
        
        # GradCAM heatmap image
        grad_heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam_map.squeeze().cpu()), cv2.COLORMAP_JET)
        
        # original image
        img = video[:,:,-1,:,:].squeeze().permute(1,2,0).numpy()

        # add two image
        grad_result = grad_heatmap + img
        grad_result = grad_result / np.max(grad_result)
        grad_result = np.uint8(255 * grad_result)

        if title:
            ax1.set_title('Original - {}'.format(title))
            ax2.set_title('GradCAM - {}'.format(title))
        else:
            ax1.set_title('Original')
            ax2.set_title('GradCAM')

        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 8))
        _ = ax1.imshow(img)
        _ = ax2.imshow(grad_heatmap)
        _ = ax3.imshow(grad_result)
        fig.tight_layout()
        
        if save_dir:
            fig.savefig(save_dir)
        
        return img, grad_heatmap, grad_result, fig
    
# GradCAM Wrapper for SlowFast
class GradCAM_SlowFast:
    def __init__(self, model : nn.Module):
        super().__init__()
        self.model = model
        
        # activations 
        self.feature_blobs_sn = []
        self.feature_blobs_fn = []

        # gradients
        self.backward_feature_sn = []
        self.backward_feature_fn = []

        def _hook_feature_sn(module:nn.Module, input:torch.Tensor, output:torch.Tensor):
            self.feature_blobs_sn.append(output.cpu().data)
            
        # Grad-CAM
        def _backward_hook_sn(module:nn.Module, input:torch.Tensor, output:torch.Tensor):
            self.backward_feature_sn.append(output[0])
            
        def _hook_feature_fn(module:nn.Module, input:torch.Tensor, output:torch.Tensor):
            self.feature_blobs_fn.append(output.cpu().data)
            
        # Grad-CAM
        def _backward_hook_fn(module:nn.Module, input:torch.Tensor, output:torch.Tensor):
            self.backward_feature_fn.append(output[0])
            
        # SlowNet case
        self.model._modules['encoder']._modules['slownet']._modules['layer4']._modules['0'].get_submodule("conv3").register_forward_hook(_hook_feature_sn)
        self.model._modules['encoder']._modules['slownet']._modules['layer4']._modules['0'].get_submodule("conv3").register_backward_hook(_backward_hook_sn)

        # FastNet case
        self.model._modules['encoder']._modules['fastnet'].get_submodule('l_layer3').register_forward_hook(_hook_feature_fn)
        self.model._modules['encoder']._modules['fastnet'].get_submodule('l_layer3').register_backward_hook(_backward_hook_fn)

        self.model.eval()
            
    def __call__(self, video : torch.Tensor, title : Optional[str] = None, save_dir :Optional[str] = None):
        logit = self.model(video)
        output = torch.nn.functional.softmax(logit, dim = 1).data.squeeze()
        
        score = logit[:,0].squeeze()
        score.backward(retain_graph = True)
        
        activations_sn = self.feature_blobs_sn[-1] # (1, 512, 7, 7), forward activations
        gradients_sn = self.backward_feature_sn[-1] # (1, 512, 7, 7), backward gradients

        print("SlowNet | activations : ", activations_sn.size())
        print("SlowNet | gradients : ", gradients_sn.size())

        activations_fn = self.feature_blobs_fn[-1] # (1, 512, 7, 7), forward activations
        gradients_fn = self.backward_feature_fn[-1] # (1, 512, 7, 7), backward gradients

        print("FastNet | activations : ", activations_fn.size())
        print("FastNet | gradients : ", gradients_fn.size())

        alpha = gradients_sn.mean(2) # time axis average
        alpha = alpha.view(1, 512, -1).mean(2) # average for all spatial axis
        weights_sn = alpha.view(1, 512, 1, 1, 1)

        alpha = gradients_fn.mean(2) # time axis average
        alpha = alpha.view(1, 64, -1).mean(2) # average for all spatial axis
        weights_fn = alpha.view(1, 64, 1, 1, 1)
        
        # case : SlowNet 
        grad_cam_map = (weights_sn*activations_sn).sum(1, keepdim = True)
        grad_cam_map = F.relu(grad_cam_map).view(1,1,5,4,4)

        grad_cam_map_interp = torch.empty((1,1,5,128,128))

        for idx in range(3):
            grad_cam_map_interp[:,:,idx,:,:] = F.interpolate(grad_cam_map[:,:,idx,:,:], size=(128, 128), mode='bilinear', align_corners=False)

        grad_cam_map_interp = grad_cam_map_interp.mean(2)
        map_min, map_max = grad_cam_map_interp.min(), grad_cam_map_interp.max()
        grad_cam_map_sn = (grad_cam_map_interp - map_min).div(map_max - map_min).data

        # case : FastNet 
        grad_cam_map = (weights_fn*activations_fn).sum(1, keepdim = True)
        grad_cam_map = F.relu(grad_cam_map).view(1,1,5,8,8)

        grad_cam_map_interp = torch.empty((1,1,5,128,128))

        for idx in range(3):
            grad_cam_map_interp[:,:,idx,:,:] = F.interpolate(grad_cam_map[:,:,idx,:,:], size=(128, 128), mode='bilinear', align_corners=False)

        grad_cam_map_interp = grad_cam_map_interp.mean(2)
        map_min, map_max = grad_cam_map_interp.min(), grad_cam_map_interp.max()
        grad_cam_map_fn = (grad_cam_map_interp - map_min).div(map_max - map_min).data
        
        # GradCAM heatmap image
        grad_heatmap_sn = cv2.applyColorMap(np.uint8(255 * grad_cam_map_sn.squeeze().cpu()), cv2.COLORMAP_JET)
        grad_heatmap_fn = cv2.applyColorMap(np.uint8(255 * grad_cam_map_fn.squeeze().cpu()), cv2.COLORMAP_JET)

        # original image
        img = video[:,:,-1,:,:].squeeze().permute(1,2,0).numpy()

        # add two image
        grad_result_sn= grad_heatmap_sn + img
        grad_result_sn= grad_result_sn / np.max(grad_result_sn)
        grad_result_sn= np.uint8(255 * grad_result_sn)

        grad_result_fn= grad_heatmap_fn + img
        grad_result_fn= grad_result_fn / np.max(grad_result_fn)
        grad_result_fn= np.uint8(255 * grad_result_fn)

        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 12))
        ax1.set_title('Original - {}'.format(title))
        ax2.set_title('GradCAM : SlowNet - {}'.format(title))
        ax3.set_title('GradCAM : FastNet - {}'.format(title))
        
        if title:
            ax1.set_title('Original - {}'.format(title))
            ax2.set_title('GradCAM : SlowNet - {}'.format(title))
            ax3.set_title('GradCAM : FastNet - {}'.format(title))
        else:
            ax1.set_title('Original')
            ax2.set_title('GradCAM : SlowNet')
            ax3.set_title('GradCAM : FastNet')
            
        _ = ax1.imshow(img)
        _ = ax2.imshow(grad_heatmap_sn)
        _ = ax3.imshow(grad_heatmap_fn)
        fig.tight_layout()
        
        if save_dir:
            fig.savefig(save_dir)
        
        return img, grad_heatmap_sn, grad_heatmap_fn, fig
    
    