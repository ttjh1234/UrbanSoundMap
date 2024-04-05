from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import LinearSVR
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import argparse
from utils.metric import *
import time
import math
import h5py
import joblib 
from dataset.sound_exp import choose_region, parsing_index

region_dict= {}

region_dict[0] = [[740,1895],[1285,2351]]
region_dict[1] = [[1285,1895],[1830,2351]]
region_dict[2] = [[1830,1895],[2375,2351]]
region_dict[3] = [[2375,1895],[2920,2351]]

region_dict[4] = [[740,1439],[1285,1895]]
region_dict[5] = [[1285,1439],[1830,1895]]
region_dict[6] = [[1830,1439],[2375,1895]]
region_dict[7] = [[2375,1439],[2920,1895]]

region_dict[8] = [[195,982],[740,1439]]
region_dict[9] = [[740,982],[1285,1439]]
region_dict[10] = [[1285,982],[1830,1439]]
region_dict[11] = [[1830,982],[2375,1439]]
region_dict[12] = [[2375,982],[2920,1439]]

region_dict[13] = [[195,526],[740,982]]
region_dict[14] = [[740,526],[1285,982]]
region_dict[15] = [[1285,526],[1830,982]]
region_dict[16] = [[1830,526],[2375,982]]
region_dict[17] = [[2375,526],[2920,982]]

region_dict[18] = [[740,70],[1285,526]]
region_dict[19] = [[1285,70],[1830,526]]
region_dict[20] = [[1830,70],[2375,526]]
region_dict[21] = [[2375,70],[2920,526]]

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def parse_option():

    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--method', type=str, default='sigle', choices=['single','two'],help='Method')
    parser.add_argument('--seed', type=int, default=0,help='seed')

    opt = parser.parse_args()
    
    return opt

def main():
    opt = parse_option()
    
    tcord= np.load('./assets/newdata/train_coord.npy')
    vcord= np.load('./assets/newdata/test_coord.npy')
    #remove_ind=np.load('./assets/newdata/mlremoveind.npy')

    cord = np.concatenate([tcord, vcord],axis=0)
    
    t_target= np.load('./assets/newdata/train_label.npy')
    v_target= np.load('./assets/newdata/test_label.npy')
    target = np.concatenate([t_target, v_target],axis=0)
    del t_target
    del v_target

    if opt.method == 'single':
        t_img = np.load('./assets/newdata/train_img10.npy')
        v_img = np.load('./assets/newdata/test_img10.npy')
        img = np.concatenate([t_img,v_img],axis=0)
        del t_img
        del v_img
        set_random_seed(opt.seed)
        tr_idx, va_idx = choose_region(opt.seed)
        trainset, train_label, _ = parsing_index(img, target, cord, tr_idx)
        validset, valid_label, _ = parsing_index(img, target, cord, va_idx)
        trainset = trainset.reshape(train_label.shape[0],-1)
        validset = validset.reshape(valid_label.shape[0],-1)
        del img
        del target
        del cord
        del tr_idx
        del va_idx

    elif opt.method == 'two':
        t_img = np.load('./assets/newdata/train_img10.npy')
        v_img = np.load('./assets/newdata/test_img10.npy')
        t_img1 = np.load('./assets/newdata/train_img1.npy')
        v_img1 = np.load('./assets/newdata/test_img1.npy')
        img10 = np.concatenate([t_img,v_img],axis=0)
        del t_img
        del v_img
        img1 = np.concatenate([t_img1,v_img1],axis=0)
        del t_img1
        del v_img1
        set_random_seed(opt.seed)
        tr_idx, va_idx = choose_region(opt.seed)
        trainset1,trainset10, train_label,_ = parsing_index(img10, target, cord, tr_idx, img1)
        validset1,validset10, valid_label,_ = parsing_index(img10, target, cord, va_idx, img1)
        del target
        del img10
        del img1
        del cord
        del tr_idx
        del va_idx
        trainset1 = trainset1.reshape(train_label.shape[0],-1)
        validset1 = validset1.reshape(valid_label.shape[0],-1)
        trainset10 = trainset10.reshape(train_label.shape[0],-1)
        validset10 = validset10.reshape(valid_label.shape[0],-1)
        trainset = np.concatenate([trainset1,trainset10],axis=1)
        validset = np.concatenate([validset1,validset10],axis=1)
        del trainset1
        del trainset10
        del validset1
        del validset10
    else:
        raise NotImplementedError()

    result_dict = {}
    for mname in ['linear','cb','dt','xgb','lgbm']:
        result_dict[mname] = {}
        for met in ['rmse','nse','mae']:
            result_dict[mname][met] = 0

    
    SS = StandardScaler()
    _ = SS.fit_transform(trainset)
    ss_valid = SS.transform(validset)
    del trainset
    del validset
    

    for model_name in ['linear','dt','lgbm','xgb','cb']:    
        print("-----------------------")
        print("Model : ",model_name)
        print("Method : ",opt.method)
        print("Seed : ",opt.seed)    
        print("-----------------------")
        
        model1 = joblib.load('./assets/ml_model/{}_{}-{}.pkl'.format(model_name,opt.method,opt.seed))
        pred1 = model1.predict(ss_valid)
        
        result_dict[model_name]['rmse'] = rmse_metric(pred1, valid_label)
        result_dict[model_name]['nse'] = nse_metric(pred1, valid_label)
        result_dict[model_name]['mae'] = mae_metric(pred1, valid_label)
    
    result_base = pd.DataFrame(result_dict).T
    result_base.columns =  ['rmse','nse','mae']

    result_base.to_csv('./assets/ml_exp_result/result_{}_{}.csv'.format(opt.method,opt.seed))
        
    

if __name__ == "__main__":
    main()