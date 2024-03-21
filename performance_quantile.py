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
from four_models import model_dict as fm_dict
from dataset.sound_exp import get_detached_sound_valid_dataloaders
import random
import argparse


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--epochs', type=int, default=40, help='number of training epochs')
    parser.add_argument('--device', type=int, default=0, help='GPU number')
    parser.add_argument('--num_workers', type=int, default=0, help='Num Worker')
    parser.add_argument('--multigpu', type=bool, default=False, help='multigpu')
    
    # dataset
    parser.add_argument('--model', type=str, default='vgg13',
                        choices=['resnet18','vgg8','vgg11','vgg13','vgg16','vgg19','wrn_16_2','wrn_40_2',
                                 'vgg8act','vgg11act','vgg13act','vgg16act','vgg19act',
                                 'resnet8','resnet14','resnet20','resnet32',
                                 'vgg8fuse2','vgg11fuse2','vgg13fuse2','vgg16fuse2','vgg19fuse2',
                                 'myvgg8','myvgg11','myvgg13','myvgg16','myvgg19',
                                 'mymaskvgg8','mymaskvgg11','mymaskvgg13','mymaskvgg16','mymaskvgg19',
                                 'groupvgg8','groupvgg11','groupvgg13','groupvgg16','groupvgg19',
                                 'groupvgg8_2','groupvgg11_2','groupvgg13_2','groupvgg16_2','groupvgg19_2',])
    parser.add_argument('--dataset', type=str, default='sound', choices=['sound',], help='dataset')

    parser.add_argument('--metric',type=str, default='rmse')
    # parser.add_argument('--atype',type=str, default='total')
    # parser.add_argument('--mtype',type=str, default='single')

    # Experiment
    #parser.add_argument('--trial', type=int, default=0, help='the experiment id')

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


def rmse_metric(ypred,y):
    return np.sqrt(np.mean((y-ypred)**2))

def group_rmse_metric(ypred,y):    
    db_group = [30,40,50,60,70,80,90,100]
    old_db = 0
    result = []
    for db in db_group:
        mask = np.where(y<db,True,False) & np.where(y>=old_db,True,False)
        group_pred = ypred[mask]
        group_true = y[mask]
        result.append(rmse_metric(group_pred, group_true))
        old_db = db
            
    return result


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
    model_name_list = ['vgg13','vgg16']
    
    result_dict = {}
    for mname in model_name_list:
        result_dict[mname] = []

    if opt.metric =='rmse':
        metric = rmse_metric
    elif opt.metric =='group':
        metric = group_rmse_metric
    else:
        raise NotImplementedError()
    
    for i in range(3):
        set_random_seed(i)        
        _, val_loader, n_data= get_detached_sound_valid_dataloaders(path='./assets/newdata/',batch_size=opt.batch_size, num_workers=0,seed=i,atype='total')
        n_cls = 3
        mdict = fm_dict

        for mname in model_name_list:

            model_path = './assets/model/'+'quantile-{}-ADAM-1-{}.pt'.format(mname,i)
            model = load_teacher(mname,mdict,model_path,n_cls)

            model.eval()

            if torch.cuda.is_available():
                model = model.to(opt.device)

            pred = np.zeros((0,))
            target = np.zeros((0,))

            with torch.no_grad():
                for batch in tqdm(val_loader):
                    bimg = batch[0].to(opt.device)
                    blabel = batch[1]
                    out = model(bimg)[:,0].to('cpu').numpy()
                    pred = np.concatenate([pred, out], axis=0)
                    target = np.concatenate([target, blabel.numpy()])

            result_dict[mname].append(metric(pred,target))
        
    for mname in model_name_list:
        mean_rmse = np.round(np.mean(result_dict[mname]),3)
        std_rmse = np.round(np.std(result_dict[mname]),3)
        print("#######################################################")
        print("{} ".format(mname))
        print("mean_rmse : ", mean_rmse)
        print("std_rmse : ", std_rmse)
        print("#######################################################")
        for nseed in range(3):
            print("seed {} : ".format(nseed), result_dict[mname][nseed])
        print("#######################################################")
    
if __name__=="__main__":
    main()
    

