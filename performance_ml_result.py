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

def choose_region(seed):
    index_np = np.arange(0,22,1)
    valid_index = np.random.choice(index_np, 7, replace=False)
    train_index = np.delete(index_np,valid_index, axis=0)
    train_index = np.sort(train_index)
    valid_index = np.sort(valid_index)
    
    print("Seed : {}".format(seed))
    print("Train : ", train_index)
    print("Valid : ", valid_index)
             
    return train_index, valid_index

def parsing_index(data,label, coord, index):
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
    parser.add_argument('--resolution', type=int, default=1000, choices=[150,200,250,1000],help='resolution')
    parser.add_argument('--method', type=str, default='full', choices=['full','pool2','pool4','pool5','pool8','pool10','pool20'],help='resolution')
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

    urban_path = './assets/newdata/urbanform{}{}.h5py'.format(opt.resolution,opt.method)
    file_object = h5py.File(urban_path, 'r')
    rep_var = np.array(file_object['data'])
    
    #cord = np.delete(cord, remove_ind, axis=0)
    #target = np.delete(target, remove_ind, axis=0)
    
    result_dict = {}
    for mname in ['cb','dt','xgb','lgbm']:
        result_dict[mname] = {}
        for met in ['rmse','nmse','nse','fb','mae']:
            result_dict[mname][met] = 0


    set_random_seed(opt.seed)
    tr_idx, va_idx = choose_region(opt.seed)
    trainset, _ = parsing_index(rep_var,target, cord, tr_idx)
    validset, valid_label = parsing_index(rep_var,target, cord, va_idx)
    del rep_var
    del cord
    del target
    
    
    SS = StandardScaler()
    _ = SS.fit_transform(trainset)
    ss_valid = SS.transform(validset)
    
    del trainset

    for model_name in ['cb','dt','xgb','lgbm']:    
        print("-----------------------")
        print("Model : ",model_name)
        print("Method : ",opt.method)
        print("Seed : ",opt.seed)    
        print("-----------------------")
        
        model1 = joblib.load('./assets/ml_model/{}_1000_{}-{}.pkl'.format(model_name,opt.method,opt.seed))
        pred1 = model1.predict(ss_valid)
        
        result_dict[model_name]['rmse'] = rmse_metric(pred1, valid_label)
        result_dict[model_name]['nmse'] = nmse_metric(pred1, valid_label)
        result_dict[model_name]['nse'] = nse_metric(pred1, valid_label)
        result_dict[model_name]['fb'] = fb_metric(pred1, valid_label)
        result_dict[model_name]['mae'] = mae_metric(pred1, valid_label)
    
    result_base = pd.DataFrame(result_dict).T
    result_base.columns =  ['rmse','nmse','nse','fb','mae']

    result_base.to_csv('./assets/ml_exp_result/result_{}.csv'.format(opt.seed))
        
    

if __name__ == "__main__":
    main()