import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import LinearSVR
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import h5py
import joblib
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

    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--device', type=int, default=0, help='GPU number')
    parser.add_argument('--num_workers', type=int, default=0, help='Num Worker')
    
    # dataset
    parser.add_argument('--mname', type=str, default='linear',
                        choices=['linear','DT','LGBM','XGB','CB'])
    
    
    opt = parser.parse_args()
        
    if opt.device==0:
        opt.device='cuda:0'
    elif opt.device==1:
        opt.device='cuda:1'
    else:
        opt.device='cpu'

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

    return data[data_index], label[data_index], coord[data_index]

def main():

    opt = parse_option()
    
    # dataloader
    tcord= np.load('./assets/newdata/train_coord.npy')
    vcord= np.load('./assets/newdata/test_coord.npy')

    cord = np.concatenate([tcord, vcord],axis=0)
    del tcord
    del vcord

    t_target= np.load('./assets/newdata/train_label.npy')
    v_target= np.load('./assets/newdata/test_label.npy')
    target = np.concatenate([t_target, v_target],axis=0)
    del t_target
    del v_target

    urban_path = './assets/newdata/urbanform1000pool2.h5py'
    file_object = h5py.File(urban_path, 'r')
    rep_var = np.array(file_object['data'])
    
    set_random_seed(0)
    tr_idx, va_idx = ml_choose_region(0)
    trainset, train_label, train_coord = ml_parsing_index(rep_var,target, cord, tr_idx)
    validset, valid_label, valid_coord = ml_parsing_index(rep_var,target, cord, va_idx)
    label = np.concatenate([train_label,valid_label],axis=0)
    coord = np.concatenate([train_coord,valid_coord],axis=0)
    
    del rep_var
    del cord
    del target
    del train_label
    del valid_label
    
    SS = StandardScaler()
    ss_train = SS.fit_transform(trainset)
    del trainset
    ss_valid = SS.transform(validset)
    vdata = np.concatenate([ss_train,ss_valid],axis=0)
    del ss_train
    del ss_valid
    
 
    v_des = torch.tensor(vdata,dtype=torch.float)
    v_label = torch.tensor(label,dtype=torch.float)
    v_coord = torch.tensor(coord,dtype=torch.float)
    vdata = TensorDataset(v_des,v_label,v_coord)

    val_loader = DataLoader(vdata,batch_size=v_label.shape[0],shuffle=False,num_workers=0)
    
    model = joblib.load('./assets/ml_model/{}_1000_pool2-{}.pkl'.format(opt.mname.lower(),0))
    prediction = np.zeros((0,))
    target = np.zeros((0,))
    coord = np.zeros((0,2))
    for batch in tqdm(val_loader):
        bimg = batch[0].numpy()
        target = np.concatenate([target, batch[1].numpy()],axis=0)
        coord = np.concatenate([coord, batch[2].numpy()],axis=0)

        pred = model.predict(bimg)
        prediction = np.concatenate([prediction,pred],axis=0)
            
    np.save('./assets/vdata/total_pred_ml_{}_0'.format(opt.mname),prediction)
    np.save('./assets/vdata/total_label_ml_{}_0'.format(opt.mname),target)   
    np.save('./assets/vdata/total_coord_ml_{}_0'.format(opt.mname),coord)

if __name__ == '__main__':
    main()
