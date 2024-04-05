import torch
import numpy as np
import os
import pandas as pd
from utils.util import *
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from models import model_dict as sm_dict
from two_models import model_dict as tm_dict
from dataset.sound_exp import get_sound_valid_dataloaders, get_two_sound_valid_dataloaders
import random
import argparse
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

def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--device', type=int, default=0, help='GPU number')
    parser.add_argument('--trial', type=int, default=0, help='Seed number')
    

    parser.add_argument('--atype',type=str, default='batch',choices = ['batch','total','single'])
    parser.add_argument('--mtype',type=str, default='ml', choices = ['ml','cnn','cnn_two'])

    opt = parser.parse_args()
    
    if opt.device==0:
        opt.device='cuda:0'
    elif opt.device==1:
        opt.device='cuda:1'
    else:
        opt.device='cpu'

    return opt

def load_teacher(model_name, model_dict, model_path, n_cls):
    print('==> loading teacher model')
    model = model_dict[model_name](num_classes=n_cls)
    
    try:
        print("Single GPU Model Load")
        model.load_state_dict(torch.load(model_path))
        print("Load Single Model")
    except:
        print("Mutil GPU Model Load")
        state_dict=torch.load(model_path)
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('module.','')
            new_state_dict[new_key] = state_dict[key]
        model.load_state_dict(new_state_dict)
        print("Load Single GPU Model from Multi GPU Model")
    
    print('==> done')
    return model


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

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def main():
    opt = parse_option()
    set_random_seed(opt.trial)
    if opt.mtype == 'cnn':
        if opt.atype == 'batch':
            _, val_loader, n_data= get_sound_valid_dataloaders(path='./assets/newdata/',batch_size=opt.batch_size, num_workers=0,seed=opt.trial,atype='total')
        elif opt.atype =='single':
            _, val_loader, n_data = get_sound_valid_dataloaders(path='./assets/newdata/',batch_size=1, num_workers=0,seed=opt.trial,atype='total')
        else:
            raise NotImplementedError()
        
        n_cls = 1
        mdict = sm_dict
        
        model_name_list = ['vgg13','vgg16','resnet18','wrn_40_2']
        result_dict = {}
        for mname in model_name_list:
            result_dict[mname] = []

        for mname in model_name_list:
            model_path = './assets/aug_exp_model/'+'{}-{}-ADAM-1-{}.pt'.format('Aug_eff',mname,opt.trial)
            model = load_teacher(mname,mdict,model_path,n_cls)
            
            model.eval()

            if torch.cuda.is_available():
                model = model.to(opt.device)

            if opt.device == 'cpu':
                n=0
                with torch.no_grad():
                    for batch in tqdm(val_loader):
                        bimg = batch[0].to(opt.device)
                        start = time.perf_counter()
                        _ = model(bimg)
                        end = time.perf_counter()
                        temp_time = end-start
                        print(f"Elapsed time on CPU : {temp_time} seconds")
                        result_dict[mname].append(temp_time)
                        n+=1
                        if n == 200:
                            break
                
            elif opt.device == 'cuda:0':
                n=0
                with torch.no_grad():
                    for batch in tqdm(val_loader):
                        start_event = torch.cuda.Event(enable_timing = True)
                        end_event = torch.cuda.Event(enable_timing = True)
                        bimg = batch[0].to(opt.device)
                        start_event.record()
                        _ = model(bimg)
                        end_event.record()
                        torch.cuda.synchronize()
                        temp_time = start_event.elapsed_time(end_event)    
                        print(f"Elapsed time on GPU : {temp_time * 1e-3} seconds")
                        result_dict[mname].append(temp_time)
                        n+=1
                        if n == 200:
                            break
            else:
                raise NotImplementedError()

        result_time = pd.DataFrame(result_dict)
        result_time.to_csv('./assets/ml_exp_result/time_speed_cnn_{}_{}.csv'.format(opt.atype,opt.device))
        
    elif opt.mtype == 'cnn_two':
        if opt.atype == 'batch':
            _, val_loader, n_data= get_two_sound_valid_dataloaders(path='./assets/newdata/',batch_size=opt.batch_size, num_workers=0,seed=opt.trial,atype='total')
        elif opt.atype =='single':
            _, val_loader, n_data = get_two_sound_valid_dataloaders(path='./assets/newdata/',batch_size=1, num_workers=0,seed=opt.trial,atype='total')
        else:
            raise NotImplementedError()
        
        n_cls = 1
        mdict = tm_dict
        
        model_name_list = ['vgg13','vgg16','resnet18','wrn_40_2']
        result_dict = {}
        for mname in model_name_list:
            result_dict[mname] = []

        for mname in model_name_list:
            model_path = './assets/model/'+'{}-{}-ADAM-1-{}.pt'.format('two',mname,opt.trial)
            model = load_teacher(mname,mdict,model_path,n_cls)
            
            model.eval()

            if torch.cuda.is_available():
                model = model.to(opt.device)

            if opt.device == 'cpu':
                n=0
                with torch.no_grad():
                    for batch in tqdm(val_loader):
                        bimg = batch[0].to(opt.device)
                        start = time.perf_counter()
                        _ = model(bimg)
                        end = time.perf_counter()
                        temp_time = end-start
                        print(f"Elapsed time on CPU : {temp_time} seconds")
                        result_dict[mname].append(temp_time)
                        n+=1
                        if n == 200:
                            break
                
            elif opt.device == 'cuda:0':
                n=0
                with torch.no_grad():
                    for batch in tqdm(val_loader):
                        start_event = torch.cuda.Event(enable_timing = True)
                        end_event = torch.cuda.Event(enable_timing = True)
                        bimg = batch[0].to(opt.device)
                        start_event.record()
                        _ = model(bimg)
                        end_event.record()
                        torch.cuda.synchronize()
                        temp_time = start_event.elapsed_time(end_event)    
                        print(f"Elapsed time on GPU : {temp_time * 1e-3} seconds")
                        result_dict[mname].append(temp_time)
                        n+=1
                        if n == 200:
                            break
            else:
                raise NotImplementedError()

        result_time = pd.DataFrame(result_dict)
        result_time.to_csv('./assets/ml_exp_result/time_speed_cnn_two_{}_{}.csv'.format(opt.atype,opt.device))
        
    elif opt.mtype == 'ml':
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
        
        set_random_seed(opt.trial)
        tr_idx, va_idx = ml_choose_region(opt.trial)
        trainset, train_label = ml_parsing_index(rep_var,target, cord, tr_idx)
        validset, valid_label = ml_parsing_index(rep_var,target, cord, va_idx)
        del rep_var
        del cord
        del target
        del train_label
        
        SS = StandardScaler()
        _ = SS.fit_transform(trainset)
        del trainset
        ss_valid = SS.transform(validset)
        
        #model_name_list = ['DT','LGBM','XGB','CB']
        model_name_list = ['linear']
        result_dict = {}
        for mname in model_name_list:
            result_dict[mname] = []
        v_des = torch.tensor(ss_valid,dtype=torch.float)
        v_label = torch.tensor(valid_label,dtype=torch.float)
        vdata = TensorDataset(v_des,v_label)
        if opt.atype =='batch':
            val_loader = DataLoader(vdata,batch_size=128,shuffle=False,num_workers=0)
        elif opt.atype == 'single':
            val_loader = DataLoader(vdata,batch_size=1,shuffle=False,num_workers=0)
        else:
            raise NotImplementedError()

        for mname in model_name_list:
            model = joblib.load('./assets/ml_model/{}_1000_pool2-{}.pkl'.format(mname.lower(),opt.trial))
            n=0
            for batch in tqdm(val_loader):
                bimg = batch[0].numpy()
                start = time.perf_counter()
                _ = model.predict(ss_valid)
                end = time.perf_counter()
                temp_time = end-start
                print(f"Elapsed time on CPU : {temp_time} seconds")
                result_dict[mname].append(temp_time)
                n+=1
                if n == 200:
                    break

        result_time = pd.DataFrame(result_dict)
        result_time.to_csv('./assets/ml_exp_result/time_speed_linear_{}_cpu.csv'.format(opt.atype))

    else:
        raise NotImplementedError()
    
    
if __name__=="__main__":
    main()