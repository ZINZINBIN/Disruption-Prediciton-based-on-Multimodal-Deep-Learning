import numpy as np
import torch 
import torch.nn as nn
from typing import Tuple, List, Dict, Optional
from pytorch_model_summary import summary
from abc import ABCMeta, abstractclassmethod, abstractmethod, abstractstaticmethod

class AbstractEncoder(metaclass = ABCMeta):

    tag = "Image Encoder"

    @abstractmethod
    def forward(self):
        pass
    
    @abstractmethod
    def get_output_shape(self):
        pass

    @abstractmethod
    def show_strucuture(self):
        pass

    @abstractmethod
    def show_CAM(self):
        pass

    @abstractmethod
    def show_Grad_CAM(self):
        pass

    @abstractmethod
    def device_allocation(self, device : str):
        pass
    
class AbstractClassifier(metaclass = ABCMeta):

    tag = "Classifier"

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def show_strucuture(self):
        pass

    @abstractmethod
    def device_allocation(self, device : str):
        pass

class AbstractModel(metaclass = ABCMeta):

    tag = "Disruption Predictor"

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass
    
    @abstractmethod
    def save_weight(self):
        pass
    
    @abstractmethod
    def show_structure(self):
        pass

    @abstractmethod
    def device_allocation(self, device : str):
        pass