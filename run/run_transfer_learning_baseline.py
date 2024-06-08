import argparse
import time
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import math

import wandb
from models import model_dict as sm_dict
from two_models import model_dict as tm_dict
from utils.loop import train, evaluate
from utils.util import epoch_time, adjust_learning_rate
from dataset.sound_ood import get_ood_two_sound_dataloaders, get_ood_sound_dataloaders
from dataset.sound_da2 import get_sample_ood_two_sound_dataloaders,  get_sample_ood_sound_dataloaders

def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--epochs', type=int, default=40, help='number of training epochs')
    parser.add_argument('--device', type=int, default=0, help='GPU number')
    parser.add_argument('--num_workers', type=int, default=0, help='Num Worker')
    parser.add_argument('--multigpu', type=bool, default=False, help='multigpu')
    
    # optimization
    parser.add_argument('--optimizer', type=str, default="ADAM", help='Optimizer', choices=["ADAM", "ADAMW"])
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='10,20,30', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.5, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay') # default 5e-4
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--method', type=str, default='single',
                        choices=['single','two','multi','four'], help = 'Channel Architecture')    
    parser.add_argument('--model', type=str, default='vgg13',
                        choices=['vgg8','vgg11','vgg13','vgg16','vgg19','resnet18','wrn_16_2','wrn_40_2'])
    parser.add_argument('--dataset', type=str, default='DJ', choices=['DJ','Seoul'], help='dataset')
    parser.add_argument('--n_layer', type=str, default='last',choices=['last','full'], help = 'Select Fine Tuning Layer')
    parser.add_argument('--d_num', type=int, default=10, help = 'The number of Training Number')
    # Experiment
    parser.add_argument('--trial', type=int, default=0, help='the experiment id')
    parser.add_argument('--run_flag', type=int, default=0, help='run_flag')    

    opt = parser.parse_args()
    
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))
    
    if opt.device==0:
        opt.device='cuda:0'
    elif opt.device==1:
        opt.device='cuda:1'
    else:
        opt.device='cpu'

    return opt

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

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


def main():

    opt = parse_option()
    
    set_random_seed(opt.trial)

    if opt.method == 'two':
        mean_v = np.load('./assets/sound_map/mean_vec.npy')
        std_v = np.load('./assets/sound_map/std_vec.npy')

        valid_loader = get_ood_two_sound_dataloaders(path='./assets/newdata/', region = opt.dataset, mean_vec=mean_v, std_vec=std_v,
                                                            batch_size=512, num_workers=opt.num_workers)
        n_cls = 1
        mdict = tm_dict
        file_path = './assets/sound_map/'+'two-{}-ADAM-1-0.pt'.format(opt.model)
    elif opt.method =='single':
        mean_v = np.load('./assets/sound_map/single_mean_vec.npy')
        std_v = np.load('./assets/sound_map/single_std_vec.npy')

        valid_loader = get_ood_sound_dataloaders(path='./assets/newdata/', region = opt.dataset, mean_vec=mean_v, std_vec=std_v,
                                                 batch_size=512, num_workers=opt.num_workers)
        n_cls = 1
        mdict = sm_dict
        file_path = './assets/sound_map/'+'single-{}-ADAM-1-0.pt'.format(opt.model)

    else:
        raise NotImplementedError(opt.dataset)

    # model
    model = load_teacher(opt.model,mdict,file_path,n_cls)
    model.eval()

    criterion = nn.MSELoss()

    if torch.cuda.is_available():
        model = model.to(opt.device)
        criterion = criterion.to(opt.device)
        
    # routine
    run=0
    valid_loss, valid_rmse = evaluate(model, valid_loader, criterion, opt.device, run)

    print("{}-{}-{} : ".format(opt.model, opt.dataset, opt.method),valid_rmse)            
    


if __name__ == '__main__':
    main()