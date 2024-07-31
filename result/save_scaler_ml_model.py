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
from utils.util import epoch_time
import time
import math
import h5py
import joblib 
from dataset.sound import choose_region, parsing_index
from utils.util import set_random_seed

# Region Dict : For Gwangju dataset, We split whole region into non-overlap grid. 
# For dividing data to train and validation set, define coordination of each grid.
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

def parse_option():

    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--method', type=str, default='single', choices=['single','two','F'],help='Method')
    parser.add_argument('--seed', type=int, default=0,help='seed')

    opt = parser.parse_args()
    
    return opt

def ml_choose_region(seed):
    index_np = np.arange(0,22,1)
    valid_index = np.random.choice(index_np, 7, replace=False)
    train_index = np.delete(index_np,valid_index, axis=0)
    train_index = np.sort(train_index)
    valid_index = np.sort(valid_index)
    
    print("Seed : {}".format(seed))
    print("Train : ", train_index)
    print("Valid : ", valid_index)
             
    return train_index, valid_index

def ml_parsing_index(data,label, coord, index):
    data_index = np.zeros((0,)).astype(int)
    for i in range(index.shape[0]):
        min_coord, max_coord = region_dict[index[i]]
        min_x, min_y = min_coord
        max_x, max_y = max_coord
        if max_y == 2351:
            if max_x == 2920:
                region_mask = ((coord[:,0]>=min_x) & (coord[:,0]<=max_x)) & ((coord[:,1]>=min_y) & (coord[:,1]<=max_y))
            else:
                region_mask = ((coord[:,0]>=min_x) & (coord[:,0]<max_x)) & ((coord[:,1]>=min_y) & (coord[:,1]<=max_y))
        else:
            if max_x == 2920:
                region_mask = ((coord[:,0]>=min_x) & (coord[:,0]<=max_x)) & ((coord[:,1]>=min_y) & (coord[:,1]<max_y))
            else:
                region_mask = ((coord[:,0]>=min_x) & (coord[:,0]<max_x)) & ((coord[:,1]>=min_y) & (coord[:,1]<max_y))
                
        data_index = np.concatenate([data_index, np.where(region_mask)[0]],axis=0)

    return data[data_index], label[data_index]

def main():
    
    opt = parse_option()
    
    tcord= np.load('./assets/newdata/train_coord.npy')
    vcord= np.load('./assets/newdata/test_coord.npy')
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
        
    elif opt.method=='F':
        urban_path = './assets/newdata/urbanform1000pool2.h5py'
        file_object = h5py.File(urban_path, 'r')
        rep_var = np.array(file_object['data'])
        
        set_random_seed(0)
        tr_idx, va_idx = ml_choose_region(0)
        trainset, train_label = ml_parsing_index(rep_var,target, cord, tr_idx)
        del rep_var
        del cord
        del target
        del train_label
        
    else:
        raise NotImplementedError()

    print("-----------------------")
    print("Method : ",opt.method)
    print("Seed : ",opt.seed)    
    print("-----------------------")

    
    SS = StandardScaler()
    _ = SS.fit_transform(trainset)
    
    joblib.dump(SS, './assets/ml_model/scaler_{}_0.pkl'.format(opt.method))
    ml_mean = SS.mean_
    ml_std = SS.scale_
    np.save('./assets/newdata/ml_{}_mean'.format(opt.method), ml_mean)
    np.save('./assets/newdata/ml_{}_std'.format(opt.method), ml_std)
    
    
if __name__=="__main__":
    main()