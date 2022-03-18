import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    return np.mean((pred - true)**2)

def Corr(pred, true):
    sig_p = np.std(pred, axis = 0)
    sig_g = np.std(true, axis = 0)
    m_p = pred.mean(0)
    m_g = true.mean(0)
    int = (sig_g != 0)