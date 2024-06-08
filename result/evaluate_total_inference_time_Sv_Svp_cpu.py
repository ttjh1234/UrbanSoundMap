import argparse
import time
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import math
import os
from utils.util import *
from torch.utils.data import TensorDataset, DataLoader

import random
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import h5py
import joblib
from dataset.sound_hdf5 import get_ml_sv_total_dataloaders, get_ml_svp_total_dataloaders

def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--trial', type=int, default=0, help='Seed number')
    parser.add_argument('--mname', type=str,default='linear', choices=['linear','dt','lgbm','xgb','cb'])
    parser.add_argument('--method', type=str,default='single', choices=['single','two'])
    parser.add_argument('--num_workers', type=int, default=0, help='Seed number')
    
    opt = parser.parse_args()

    return opt


def main():
    start = time.perf_counter()
    opt = parse_option()
    start_data = time.perf_counter()
    if opt.method == 'single':
        test_loader, num_data = get_ml_sv_total_dataloaders(path = './assets/', batch_size= opt.batch_size, num_workers=opt.num_workers, seed=opt.trial)
        model = joblib.load('./assets/ml_model/{}_single-{}.pkl'.format(opt.mname,opt.trial))
    elif opt.method == 'two':
        test_loader, num_data = get_ml_svp_total_dataloaders(path = './assets/', batch_size= opt.batch_size, num_workers=opt.num_workers, seed=opt.trial)
        model = joblib.load('./assets/ml_model/{}_two-{}.pkl'.format(opt.mname,opt.trial))
    end_data = time.perf_counter()
    
    start_process = time.perf_counter()
    temp_time = 0
    for batch in tqdm(test_loader):
        start_event = time.perf_counter()
        _ = model.predict(batch.numpy())
        end_event = time.perf_counter()
        temp_time += (end_event-start_event)

    end_process = time.perf_counter()
    end = time.perf_counter()
    total_time = end-start
    total_data = end_data - start_data
    total_process = end_process - start_process
    print("==================================================================")
    print("{}".format(opt.method))
    print("{}-{} Total Batch Processing Time (Not Consider Memory Loading) :".format(opt.mname,opt.batch_size), temp_time)
    print("{}-{} Time upto DataLoader :".format(opt.mname,opt.batch_size), total_data)
    print("{}-{} Total Batch Processing Time (Batch Processing Time + Memory Loading) :".format(opt.mname,opt.batch_size), total_process)
    print("{}-{} Total Processing Time :".format(opt.mname,opt.batch_size), total_time)
    print("{}-{} Total Memory Loading Time :".format(opt.mname,opt.batch_size), total_time - temp_time)
    print("==================================================================")
    
if __name__ == '__main__':
    main()
