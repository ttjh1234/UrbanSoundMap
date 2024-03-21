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
from multi_models import model_dict as mm_dict
from four_models import model_dict as fm_dict
from two_models import model_dict as tm_dict
from dataset.sound_exp import get_mean_std_datached, get_mean_std_datached_origin, get_mean_std_single, get_mean_std_two
from dataset.sound_ood import get_ood_detached_origin_sound_dataloaders, get_ood_detached_sound_dataloaders, get_ood_sound_dataloaders, get_ood_two_sound_dataloaders
import random
import argparse


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--device', type=int, default=0, help='GPU number')
    parser.add_argument('--num_workers', type=int, default=0, help='Num Worker')
    parser.add_argument('--region', type=str, default = 'DJ', choices=['DJ','Seoul'])
    parser.add_argument('--multigpu', type=bool, default=False, help='multigpu')
    
    parser.add_argument('--mtype',type=str, default='single')

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
    model_name_list = ['vgg13','vgg16','resnet18','wrn_40_2']
    
    result_dict = {}
    for mname in model_name_list:
        result_dict[mname] = []

    
    for i in range(10):
        set_random_seed(i)
        if opt.mtype == 'single':
            mean_v, std_v = get_mean_std_single(path='./assets/newdata/',seed = i)
            test_loader = get_ood_sound_dataloaders(path='./assets/newdata/', region = opt.region, mean_vec=mean_v, std_vec=std_v,
                                                             batch_size=opt.batch_size, num_workers=opt.num_workers)
            n_cls = 1
            mdict = sm_dict
        elif opt.mtype == 'two':
            mean_v, std_v = get_mean_std_two(path='./assets/newdata/',seed = i)
            test_loader = get_ood_two_sound_dataloaders(path='./assets/newdata/', region = opt.region, mean_vec=mean_v, std_vec=std_v,
                                                             batch_size=opt.batch_size, num_workers=opt.num_workers)
            n_cls = 1
            mdict = tm_dict
        elif opt.mtype == 'multi':
            mean_v, std_v = get_mean_std_datached_origin(path='./assets/newdata/',seed = i)
            test_loader = get_ood_detached_origin_sound_dataloaders(path='./assets/newdata/', region = opt.region, mean_vec=mean_v, std_vec=std_v,
                                                             batch_size=opt.batch_size, num_workers=opt.num_workers)
            n_cls = 1
            mdict = mm_dict
        elif opt.mtype == 'four':
            mean_v, std_v = get_mean_std_datached(path='./assets/newdata/',seed = i)
            test_loader = get_ood_detached_sound_dataloaders(path='./assets/newdata/', region = opt.region, mean_vec=mean_v, std_vec=std_v,
                                                             batch_size=opt.batch_size, num_workers=opt.num_workers)
            n_cls = 1
            mdict = fm_dict
        else:
            raise NotImplementedError()

        for mname in model_name_list:
            if opt.mtype == 'single':
                model_path = './assets/aug_exp_model/'+'{}-{}-ADAM-1-{}.pt'.format('Aug_eff',mname,i)
                model = load_teacher(mname,mdict,model_path,n_cls)
            elif opt.mtype =='two':
                model_path = './assets/model/'+'{}-{}-ADAM-1-{}.pt'.format('two',mname,i)
                model = load_teacher(mname,mdict,model_path,n_cls)
            elif opt.mtype =='multi':
                model_path = './assets/model/'+'{}-{}-ADAM-1-{}.pt'.format('decoupled',mname,i)
                model = load_teacher(mname,mdict,model_path,n_cls)
            elif opt.mtype =='four':
                model_path = './assets/model/'+'{}-{}-ADAM-1-{}.pt'.format('four',mname,i)
                model = load_teacher(mname,mdict,model_path,n_cls)
                
            if opt.multigpu:
                model =nn.DataParallel(model,device_ids=[0,1])

            if torch.cuda.is_available():
                model = model.to(opt.device)

            model.eval()
            pred = np.zeros((0,))
            target = np.zeros((0,))

            with torch.no_grad():
                for batch in tqdm(test_loader):
                    bimg = batch[0].to(opt.device)
                    blabel = batch[1]
                    out = model(bimg).squeeze(1).to('cpu').numpy()
                    pred = np.concatenate([pred, out], axis=0)
                    target = np.concatenate([target, blabel.numpy()])

            result_dict[mname].append(rmse_metric(pred,target))
        
    result_csv = pd.DataFrame(result_dict).T
    mean_vec = result_csv.mean(axis=1)
    std_vec = result_csv.std(axis=1)
    result_csv['mean'] = mean_vec
    result_csv['std'] = std_vec
    
    result_csv.to_csv('./assets/ood_file/ood_{}_{}.csv'.format(opt.region, opt.mtype))
    
if __name__=="__main__":
    main()
    

