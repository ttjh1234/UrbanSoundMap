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
from dataset.sound_ood import get_ood_two_sound_dataloaders,get_ood_sound_dataloaders,get_ood_detached_sound_dataloaders, get_ood_detached_origin_sound_dataloaders
from dataset.sound_da import get_sample_ood_two_sound_dataloaders,get_sample_ood_sound_dataloaders,get_sample_ood_detached_sound_dataloaders, get_sample_ood_detached_origin_sound_dataloaders
from dataset.sound_exp import get_mean_std_datached, get_mean_std_datached_origin, get_mean_std_single, get_mean_std_two

def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--epochs', type=int, default=30, help='number of training epochs') # 60
    parser.add_argument('--device', type=int, default=0, help='GPU number')
    parser.add_argument('--num_workers', type=int, default=0, help='Num Worker')
    parser.add_argument('--multigpu', type=bool, default=False, help='multigpu')
    
    # optimization
    parser.add_argument('--optimizer', type=str, default="ADAM", help='Optimizer', choices=["ADAM", "ADAMW"])
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='10,20', help='where to decay lr, can be a list') # 10,20,30
    parser.add_argument('--lr_decay_rate', type=float, default=0.05, help='decay rate for learning rate')
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
    
    if opt.run_flag==1:
        wandb.init(
            project="sound_prediction_finetune",
            name="{}-{}-{}-{}-{}-{}-{}-{}-{}-Baseline".format(opt.dataset,opt.method,opt.model,opt.batch_size,opt.n_layer,opt.d_num,opt.optimizer,int(100000*opt.learning_rate),opt.trial),
            config={
                "optimizer" : opt.optimizer,
                "learning_rate" : opt.learning_rate,
                "model" : opt.model,
                "trial_num" : opt.trial,
                "dataset" : opt.dataset,
                "method" : opt.method,
                "n_layer" : opt.n_layer,
                "number of sample" : opt.d_num, 
            }
        )
        run=1
    else:
        run=0
    
    set_random_seed(opt.trial)

    
    if opt.method == 'single':
        mean_v, std_v = get_mean_std_single(path='./assets/newdata/',seed = opt.trial)
        train_loader = get_sample_ood_sound_dataloaders(path='./assets/newdata/', region = opt.dataset,sample_size=opt.d_num, mean_vec=mean_v, std_vec=std_v,
                                                            batch_size=opt.batch_size, num_workers=opt.num_workers)

        valid_loader = get_ood_sound_dataloaders(path='./assets/newdata/', region = opt.dataset, mean_vec=mean_v, std_vec=std_v,
                                                            batch_size=512, num_workers=opt.num_workers)
        n_cls = 1
        mdict = sm_dict
        file_path = './assets/aug_exp_model/'+'{}-{}-ADAM-1-{}.pt'.format('Aug_eff',opt.model,opt.trial)

    elif opt.method == 'two':
        mean_v, std_v = get_mean_std_two(path='./assets/newdata/',seed = opt.trial)
        train_loader = get_sample_ood_two_sound_dataloaders(path='./assets/newdata/', region = opt.dataset, sample_size=opt.d_num, mean_vec=mean_v, std_vec=std_v,
                                                            batch_size=opt.batch_size, num_workers=opt.num_workers)

        valid_loader = get_ood_two_sound_dataloaders(path='./assets/newdata/', region = opt.dataset, mean_vec=mean_v, std_vec=std_v,
                                                            batch_size=512, num_workers=opt.num_workers)
        n_cls = 1
        mdict = tm_dict
        file_path = './assets/model/'+'{}-{}-ADAM-1-{}.pt'.format('two',opt.model,opt.trial)

    elif opt.method == 'multi':
        mean_v, std_v = get_mean_std_datached_origin(path='./assets/newdata/',seed = opt.trial)
        train_loader = get_sample_ood_detached_origin_sound_dataloaders(path='./assets/newdata/', region = opt.dataset,sample_size = opt.d_num, mean_vec=mean_v, std_vec=std_v,
                                                            batch_size=opt.batch_size, num_workers=opt.num_workers)
   
        valid_loader = get_ood_detached_origin_sound_dataloaders(path='./assets/newdata/', region = opt.dataset, mean_vec=mean_v, std_vec=std_v,
                                                            batch_size=512, num_workers=opt.num_workers)
        n_cls = 1
        mdict = mm_dict
        file_path = './assets/model/'+'{}-{}-ADAM-1-{}.pt'.format('decoupled',opt.model,opt.trial)

    elif opt.method == 'four':
        mean_v, std_v = get_mean_std_datached(path='./assets/newdata/',seed = opt.trial)
        train_loader = get_sample_ood_detached_sound_dataloaders(path='./assets/newdata/', region = opt.dataset,sample_size=opt.d_num, mean_vec=mean_v, std_vec=std_v,
                                                            batch_size=opt.batch_size, num_workers=opt.num_workers)

        valid_loader = get_ood_detached_sound_dataloaders(path='./assets/newdata/', region = opt.dataset, mean_vec=mean_v, std_vec=std_v,
                                                            batch_size=512, num_workers=opt.num_workers)
        n_cls = 1
        mdict = fm_dict
        file_path = './assets/model/'+'{}-{}-ADAM-1-{}.pt'.format('four',opt.model,opt.trial)

    else:
        raise NotImplementedError(opt.dataset)

    # model
    model = load_teacher(opt.model,mdict,file_path,n_cls)
    

    # optimizer
    best_loss = 1e+7

    if opt.optimizer == "ADAM":
        if opt.n_layer == 'last':
            # All parameter Freeze
            for param in model.parameters():
                param.requires_grad = False
            
            for param in model.classifier.parameters():
                param.requires_grad = True
            
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=opt.learning_rate,
                        weight_decay=opt.weight_decay)
            
        elif opt.n_layer == 'full':
            optimizer = optim.Adam(model.parameters(),
                        lr=opt.learning_rate,
                        weight_decay=opt.weight_decay)
        else:
            raise NotImplementedError()


    elif opt.optimizer == "ADAMW":
        if opt.n_layer == 'last':
            for param in model.parameters():
                param.requires_grad = False
            
            for param in model.classifier.parameters():
                param.requires_grad = True
                
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=opt.learning_rate,
                        weight_decay=opt.weight_decay)

        elif opt.n_layer == 'full':
            optimizer = optim.AdamW(model.parameters(),
                        lr=opt.learning_rate,
                        weight_decay=opt.weight_decay)

        else:
            raise NotImplementedError()
        
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
            torch.save(model.state_dict(), './assets/model_finetune/{}-{}-{}-{}-{}-{}-{}-{}-{}.pt'.format(opt.dataset,
                                                                                                       opt.method,
                                                                                                       opt.model,
                                                                                                       opt.batch_size,
                                                                                                       opt.n_layer,
                                                                                                       opt.d_num,
                                                                                                       opt.optimizer,
                                                                                                       int(100000*opt.learning_rate),
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