import numpy as np
import pandas as pd
import math


# RMSE Metric : Root Mean Square Error
def rmse_metric(ypred,y):
    return np.sqrt(np.mean((y-ypred)**2))

# NMSE Metric : Normalized Mean Square Error
def nmse_metric(ypred,y):
    true_mean = np.mean(y)
    pred_mean = np.mean(ypred)
    nmse = (np.mean((y-ypred)**2))/(true_mean * pred_mean)
    
    return nmse

# NSE Metric : Nash Sutcliffe Efficiency 
def nse_metric(ypred,y):
    ymean = np.mean(y)
    nse = 1 - (np.mean((y-ypred)**2)/np.mean((y-ymean)**2))
    return nse 

# FB Metric : Fractional mean bias 
def fb_metric(ypred,y):
    true_mean = np.mean(y)
    pred_mean = np.mean(ypred)
    fb = (2*np.mean(y - ypred)) / (true_mean + pred_mean)
    return fb

# MAE Metric : Mean Absolute Error
def mae_metric(ypred,y):
    return np.mean(np.abs(y-ypred))

