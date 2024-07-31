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

from utils.loop import train_multiple, evaluate_multiple
from utils.util import epoch_time, adjust_learning_rate, set_random_seed
from dataset.sound_multiple import get_overlap_multiple_sound_dataloaders,get_overlap_multiple_two_sound_dataloaders

def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--epochs', type=int, default=40, help='number of training epochs')
    parser.add_argument('--device', type=int, default=0, help='GPU number')
    parser.add_argument('--num_workers', type=int, default=0, help='Num Worker')
    parser.add_argument('--multigpu', type=bool, default=False, help='multigpu')
    
    # optimization
    parser.add_argument('--optimizer', type=str, default="SGD", help='Optimizer', choices=["SGD", "ADAM", "ADAMW"])
    parser.add_argument('--learning_rate', type=float, default=0.003, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='10,20,30', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.05, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay') # default 5e-4
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    
    
    parser.add_argument('--multiple_factor', type=int, default = 3, choices=[3,5,7])
    parser.add_argument('--mtype', type=str, default = 'single', choices=['single','two'])
    # dataset
    parser.add_argument('--model', type=str, default='vgg13',
                        choices=['vgg8','vgg11','vgg13','vgg16','vgg19',
                                 'wrn_16_2','wrn_40_2','resnet18','resnet50','resnet34','inception_resnetv2_2','MobileNetV2'])

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


def main():

    opt = parse_option()
    
    if opt.run_flag==1:
        wandb.init(
            project="sound_overlap_multiple_prediction",
            name="{}-{}-{}-{}-{}-{}-Baseline".format(opt.mtype,
                                                     opt.multiple_factor, 
                                                     opt.model,
                                                     opt.optimizer,
                                                     int(1000*opt.learning_rate),
                                                     opt.trial),
            config={
                "Channel" : opt.mtype,
                "Factor" : opt.multiple_factor,
                "optimizer" : opt.optimizer,
                "learning_rate" : opt.learning_rate,
                "model" : opt.model,
                "trial_num" : opt.trial,
            }
        )
        run=1
    else:
        run=0
    
    set_random_seed(opt.trial)

    # dataloader
    if opt.mtype == 'single':
        train_loader, val_loader, n_data= get_overlap_multiple_sound_dataloaders(path='./assets/multiple/',
                                                                                 batch_size=opt.batch_size, 
                                                                                 num_workers= opt.num_workers,
                                                                                 seed= opt.trial, 
                                                                                 multiple_factor = opt.multiple_factor)
        n_cls = int(opt.multiple_factor **2)   
        # model
        model = sm_dict[opt.model](num_classes=n_cls)
        
    elif opt.mtype == 'two':
        train_loader, val_loader, n_data= get_overlap_multiple_two_sound_dataloaders(path='./assets/multiple/',
                                                                                     batch_size=opt.batch_size, 
                                                                                     num_workers= opt.num_workers,
                                                                                     seed= opt.trial, 
                                                                                     multiple_factor = opt.multiple_factor)
        n_cls = int(opt.multiple_factor **2)   
        # model
        model = tm_dict[opt.model](num_classes=n_cls)

    else:
        raise NotImplementedError()
    
    if opt.multigpu:
        model =nn.DataParallel(model,device_ids=[0,1])
    # optimizer
    best_loss = 1e+5
    
    optimizer = optim.Adam(model.parameters(),
                        lr=opt.learning_rate,
                        weight_decay=opt.weight_decay)

    criterion = nn.MSELoss()

    if torch.cuda.is_available():
        model = model.to(opt.device)
        criterion = criterion.to(opt.device)
    
    # routine
    for epoch in range(1, opt.epochs + 1):
        
        adjust_learning_rate(epoch, opt, optimizer)
        
        start_time = time.time()            
        train_loss, train_rmse = train_multiple(model, train_loader, optimizer, criterion, opt.device, run)
        valid_loss, valid_rmse = evaluate_multiple(model, val_loader, criterion, opt.device, run)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        print(f'Epoch: {epoch:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.4f}')
        print(f'\t Val. Loss: {valid_loss:.4f}')
        print(f'\tTrain RMSE: {train_rmse:.4f}')
        print(f'\t Val. RMSE: {valid_rmse:.4f}')
         
        if valid_rmse < best_loss:
            best_loss = valid_rmse
            torch.save(model.state_dict(), './assets/overlap_multiple_model/{}-{}-{}-{}-{}-{}.pt'.format(opt.mtype, 
                                                                                                         opt.multiple_factor,
                                                                                                         opt.model,
                                                                                                         opt.optimizer,
                                                                                                         int(1000*opt.learning_rate),
                                                                                                         opt.trial))


        if math.isnan(train_loss):
            break
        
        if opt.run_flag==1:
            wandb.log({"train/epoch_loss": train_loss,
                       "train/epoch_rmse": train_rmse, 
                        "valid/epoch_loss" : valid_loss,
                        "valid/epoch_rmse" : valid_rmse,
                        "valid/Best_RMSE" : best_loss})    
    
    wandb.finish()


if __name__ == '__main__':
    main()