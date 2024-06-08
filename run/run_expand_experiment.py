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
from multi_models import model_dict as mm_dict
from four_models import model_dict as fm_dict
from two_models import model_dict as tm_dict
from utils.loop import train, evaluate
from utils.util import epoch_time, adjust_learning_rate
from dataset.sound_expand import get_single_dataloaders, get_two_dataloaders, get_three_sound_dataloaders,get_four_sound_dataloaders

def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--epochs', type=int, default=40, help='number of training epochs') # 60
    parser.add_argument('--device', type=int, default=0, help='GPU number')
    parser.add_argument('--num_workers', type=int, default=0, help='Num Worker')
    parser.add_argument('--multigpu', type=bool, default=False, help='multigpu')
    
    # optimization
    parser.add_argument('--optimizer', type=str, default="ADAM", help='Optimizer', choices=["ADAM", "ADAMW"])
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='10,20,30', help='where to decay lr, can be a list') # 10,20,30
    parser.add_argument('--lr_decay_rate', type=float, default=0.05, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay') # default 5e-4
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset    
    parser.add_argument('--model', type=str, default='vgg13',
                        choices=['vgg8','vgg11','vgg13','vgg16','vgg19','resnet18','wrn_16_2','wrn_40_2'])
    parser.add_argument('--resolution', type=str, default='10', help = 'The number of resolution')
    # Experiment
    parser.add_argument('--trial', type=int, default=0, help='the experiment id')
    parser.add_argument('--run_flag', type=int, default=0, help='run_flag')    

    opt = parser.parse_args()
    
    iterations = opt.lr_decay_epochs.split(',')
    iter2 = opt.resolution.split(',')

    opt.resolution_list = []
    opt.lr_decay_epochs = list([])

    for it in iterations:
        opt.lr_decay_epochs.append(int(it))
    
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
    resol_name=[str(i) for i in opt.resolution_list]
    resol_name = '-'.join(resol_name)
    
    if opt.run_flag==1:
        wandb.init(
            project="sound_prediction_ablation_expand",
            name="{}-{}-{}-{}-{}-{}".format(opt.method,opt.model,resol_name,opt.optimizer,int(1000*opt.learning_rate),opt.trial),
            config={
                "optimizer" : opt.optimizer,
                "learning_rate" : opt.learning_rate,
                "model" : opt.model,
                "trial_num" : opt.trial,
                "method" : opt.method,
                "resolution_list" : resol_name, 
            }
        )
        run=1
    else:
        run=0
    
    set_random_seed(opt.trial)

    if opt.method == 'single':
        train_loader, valid_loader, ndata = get_single_dataloaders(path='./assets/newdata/', batch_size=opt.batch_size, num_workers=opt.num_workers, seed = opt.trial)
        n_cls = 1
        mdict = sm_dict

    elif opt.method == 'two':
        train_loader, valid_loader, ndata = get_two_dataloaders(path='./assets/newdata/', resolution=opt.resolution_list[0], batch_size=opt.batch_size, num_workers=opt.num_workers, seed = opt.trial)
        n_cls = 1
        mdict = tm_dict
        
    elif opt.method == 'multi':
        train_loader, valid_loader, ndata = get_three_sound_dataloaders(path='./assets/newdata/', resolution_list=opt.resolution_list[:-1], batch_size=opt.batch_size, num_workers=opt.num_workers, seed = opt.trial)
        n_cls = 1
        mdict = mm_dict
        
    elif opt.method == 'four':
        train_loader, valid_loader, ndata = get_four_sound_dataloaders(path='./assets/newdata/', resolution_list=opt.resolution_list[:-1], batch_size=opt.batch_size, num_workers=opt.num_workers, seed = opt.trial)
        n_cls = 1
        mdict = fm_dict
        
    else:
        raise NotImplementedError(opt.method)

    model = mdict[opt.model](num_classes=n_cls)
    
    if opt.multigpu:
        model =nn.DataParallel(model,device_ids=[0,1])
    # optimizer
    best_loss = 1e+5
    # model

    if opt.optimizer == "SGD":        
        optimizer = optim.SGD(model.parameters(),
                            lr=opt.learning_rate,
                            momentum = opt.momentum,
                            weight_decay=opt.weight_decay)
        
    elif opt.optimizer == "ADAM":
        optimizer = optim.Adam(model.parameters(),
                            lr=opt.learning_rate,
                            weight_decay=opt.weight_decay)

    elif opt.optimizer == "ADAMW":
        optimizer = optim.AdamW(model.parameters(),
                            lr=opt.learning_rate,
                            weight_decay=opt.weight_decay)
        
    else:
        raise NotImplementedError(opt.optimizer)

    criterion = nn.MSELoss()

    if torch.cuda.is_available():
        model = model.to(opt.device)
        criterion = criterion.to(opt.device)
        
    # routine
    for epoch in range(1, opt.epochs + 1):
        
        adjust_learning_rate(epoch, opt, optimizer)
        
        start_time = time.time()
        train_loss, train_rmse = train(model, train_loader, optimizer, criterion, opt.device, run)
        valid_loss, valid_rmse = evaluate(model, valid_loader, criterion, opt.device, run)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        print(f'Epoch: {epoch:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.4f}')
        print(f'\t Val. Loss: {valid_loss:.4f}')
        print(f'\tTrain RMSE: {train_rmse:.4f}')
        print(f'\t Val. RMSE: {valid_rmse:.4f}')
         
        if valid_rmse < best_loss:
            best_loss = valid_rmse
            torch.save(model.state_dict(), './assets/expansion_model/{}-{}-{}-{}-{}-{}.pt'.format(opt.method,
                                                                                                  opt.model,
                                                                                                  resol_name,
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