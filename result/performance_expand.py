import torch
import numpy as np
import os
import pandas as pd
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from models import model_dict as sm_dict
from multi_models import model_dict as mm_dict
from four_models import model_dict as fm_dict
from two_models import model_dict as tm_dict
from utils.util import *
from utils.metric import rmse_metric
from dataset.sound_expand import get_four_valid_dataloaders, get_single_valid_dataloaders, get_three_valid_dataloaders, get_two_valid_dataloaders
import random
import argparse


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--device', type=int, default=0, help='GPU number')
    parser.add_argument('--num_workers', type=int, default=0, help='Num Worker')
    parser.add_argument('--multigpu', type=bool, default=False, help='multigpu')
    
    # dataset
    parser.add_argument('--resolution', type=str, default='10', help = 'The number of resolution')
    parser.add_argument('--atype',type=str, default='total')

    opt = parser.parse_args()
    
    iter2 = opt.resolution.split(',')

    opt.resolution_list = []
    
    for it in iter2:
        opt.resolution_list.append(int(it))

    if len(opt.resolution_list) == 1:
        opt.method = 'single'
    elif len(opt.resolution_list) == 2:
        opt.method = 'two'
    elif len(opt.resolution_list) == 3:
        opt.method = 'multi'
    elif len(opt.resolution_list) == 4:
        opt.method = 'four'
    else:
        raise NotImplementedError()

    if opt.device==0:
        opt.device='cuda:0'
    elif opt.device==1:
        opt.device='cuda:1'
    else:
        opt.device='cpu'

    return opt


def main():
    opt = parse_option()
    
    result_dict = {}
    result_dict['vgg13'] = []
    resol_name=[str(i) for i in opt.resolution_list]
    resol_name = '-'.join(resol_name)
    
    for i in range(5):
        set_random_seed(i)
        if opt.method == 'single':
            _, val_loader, n_data= get_single_valid_dataloaders(path='./assets/newdata/',
                                                                batch_size=256, 
                                                                num_workers=0,
                                                                seed=i,
                                                                atype=opt.atype)
            n_cls = 1
            mdict = sm_dict
        elif opt.method == 'two':
            _, val_loader, n_data= get_two_valid_dataloaders(path='./assets/newdata/',
                                                             resolution=opt.resolution_list[0], 
                                                             batch_size=256, 
                                                             num_workers=0,
                                                             seed=i,
                                                             atype=opt.atype)
            n_cls = 1
            mdict = tm_dict
        elif opt.method == 'multi':
            _, val_loader, n_data= get_three_valid_dataloaders(path='./assets/newdata/',
                                                               resolution_list=opt.resolution_list[:-1],
                                                               batch_size=256, 
                                                               num_workers=0,
                                                               seed=i,
                                                               atype=opt.atype)
            n_cls = 1
            mdict = mm_dict
        elif opt.method == 'four':
            _, val_loader, n_data= get_four_valid_dataloaders(path='./assets/newdata/',
                                                              resolution_list=opt.resolution_list[:-1],
                                                              batch_size=256, 
                                                              num_workers=0,
                                                              seed=i,
                                                              atype=opt.atype)
            n_cls = 1
            mdict = fm_dict
        else:
            raise NotImplementedError()


        if opt.method == 'two':
            model_path = './assets/expansion_model/'+'two-vgg13-{}-ADAM-1-{}.pt'.format(resol_name,i)
            model = load_teacher('vgg13',mdict,model_path,n_cls)
        elif opt.method =='multi':
            model_path = './assets/expansion_model/'+'multi-vgg13-{}-ADAM-1-{}.pt'.format(resol_name,i)
            model = load_teacher('vgg13',mdict,model_path,n_cls)
        else:
            raise NotImplementedError()
            
        model.eval()

        if torch.cuda.is_available():
            model = model.to(opt.device)

        pred = np.zeros((0,))
        target = np.zeros((0,))

        with torch.no_grad():
            for batch in tqdm(val_loader):
                bimg = batch[0].to(opt.device)
                blabel = batch[1]
                out = model(bimg).squeeze(1).to('cpu').numpy()
                pred = np.concatenate([pred, out], axis=0)
                target = np.concatenate([target, blabel.numpy()])

        result_dict['vgg13'].append(rmse_metric(pred,target))
    


    for nseed in range(5):
        print("seed {} : ".format(nseed), result_dict['vgg13'][nseed])
    
    result_csv = pd.DataFrame(result_dict).T
    mean_vec = result_csv.mean(axis=1)
    std_vec = result_csv.std(axis=1)
    result_csv['mean'] = mean_vec
    result_csv['std'] = std_vec
    
    result_csv.to_csv('./assets/paper_result_file/{}_{}_{}.csv'.format(opt.method,resol_name,opt.atype))
    
if __name__=="__main__":
    main()
    

