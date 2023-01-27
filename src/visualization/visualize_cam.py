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
from torch.autograd import Function

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