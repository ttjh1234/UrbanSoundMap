import torch
import numpy as np
import os
import pandas as pd
from utils.util import *
from utils.metric import *
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from models import model_dict as sm_dict
from multi_models import model_dict as mm_dict
from four_models import model_dict as fm_dict
from two_models import model_dict as tm_dict
from dataset.sound_exp import get_detached_sound_valid_dataloaders, get_detached_origin_sound_valid_dataloaders, get_sound_valid_dataloaders, get_two_sound_valid_dataloaders
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

    #parser.add_argument('--metric', type=str, default='rmse')
    parser.add_argument('--atype',type=str, default='total')
    parser.add_argument('--mtype',type=str, default='single')

    # Experiment
    parser.add_argument('--trial', type=int, default=0, help='the experiment id')

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
        result_dict[mname] = {}
        for met in ['rmse','nse','mae']:
            result_dict[mname][met] = 0

    set_random_seed(opt.trial)
    if opt.mtype == 'single':
        _, val_loader, n_data= get_sound_valid_dataloaders(path='./assets/newdata/',batch_size=opt.batch_size, num_workers=0,seed=opt.trial,atype=opt.atype)
        n_cls = 1
        mdict = sm_dict
    elif opt.mtype == 'two':
        _, val_loader, n_data= get_two_sound_valid_dataloaders(path='./assets/newdata/',batch_size=opt.batch_size, num_workers=0,seed=opt.trial,atype=opt.atype)
        n_cls = 1
        mdict = tm_dict
    elif opt.mtype == 'multi':
        _, val_loader, n_data= get_detached_origin_sound_valid_dataloaders(path='./assets/newdata/',batch_size=opt.batch_size, num_workers=0,seed=opt.trial,atype=opt.atype)
        n_cls = 1
        mdict = mm_dict
    elif opt.mtype == 'four':
        _, val_loader, n_data= get_detached_sound_valid_dataloaders(path='./assets/newdata/',batch_size=opt.batch_size, num_workers=0,seed=opt.trial,atype=opt.atype)
        n_cls = 1
        mdict = fm_dict
    else:
        raise NotImplementedError()

    for mname in model_name_list:
        if opt.mtype == 'single':
            model_path = './assets/aug_exp_model/'+'{}-{}-ADAM-1-{}.pt'.format('Aug_eff',mname,opt.trial)
            model = load_teacher(mname,mdict,model_path,n_cls)
        elif opt.mtype =='two':
            model_path = './assets/model/'+'{}-{}-ADAM-1-{}.pt'.format('two',mname,opt.trial)
            model = load_teacher(mname,mdict,model_path,n_cls)
        elif opt.mtype =='multi':
            model_path = './assets/model/'+'{}-{}-ADAM-1-{}.pt'.format('decoupled',mname,opt.trial)
            model = load_teacher(mname,mdict,model_path,n_cls)
        elif opt.mtype =='four':
            model_path = './assets/model/'+'{}-{}-ADAM-1-{}.pt'.format('four',mname,opt.trial)
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
                out = model(bimg).squeeze(1).to('cpu').numpy()
                pred = np.concatenate([pred, out], axis=0)
                target = np.concatenate([target, blabel.numpy()])

        result_dict[mname]['rmse'] = rmse_metric(pred, target)
        result_dict[mname]['nse'] = nse_metric(pred, target)
        result_dict[mname]['mae'] = mae_metric(pred, target)
       
        del pred
        del target
    del val_loader
            
    result_base = pd.DataFrame(result_dict).T
    result_base.columns =  ['rmse','nse','mae']

        
    result_base.to_csv('./assets/paper_result_file/result_{}_{}_{}.csv'.format(opt.atype,opt.mtype,opt.trial))
    
if __name__=="__main__":
    main()
    