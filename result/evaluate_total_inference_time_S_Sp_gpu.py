import argparse
import time
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import math
from two_models import model_dict as tm_dict
from models import model_dict as sm_dict
from dataset.sound_hdf5 import get_total_dataloaders,get_single_total_dataloaders
from utils.util import load_teacher

def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--num_workers', type=int , default=0)
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--device', type=int, default=0, help='GPU number')
    parser.add_argument('--trial', type=int, default=0, help='Seed number')
    parser.add_argument('--mtype', type=str,default='two', choices=['single','two'])
    parser.add_argument('--mname', type=str,default='vgg13', choices=['vgg13','vgg16','resnet18','wrn_40_2'])
    parser.add_argument('--multigpu', type=bool, default= False)
    
    opt = parser.parse_args()
    
    if opt.device==0:
        opt.device='cuda:0'
    elif opt.device==1:
        opt.device='cuda:1'
    else:
        opt.device='cpu'

    return opt


def main():
    start = time.perf_counter()
    opt = parse_option()
    start_data = time.perf_counter()
    if opt.mtype == 'two':
        test_loader, num_data = get_total_dataloaders(path = './assets/', batch_size= opt.batch_size, num_workers=opt.num_workers, seed=opt.trial)
        
        model_path = './assets/model/'+'{}-{}-ADAM-1-{}.pt'.format('two',opt.mname,0)
        model = load_teacher(opt.mname,tm_dict,model_path,1)
        if opt.multigpu:
            model =nn.DataParallel(model,device_ids=[0,1])
    
    elif opt.mtype == 'single':
        test_loader, num_data = get_single_total_dataloaders(path = './assets/', batch_size= opt.batch_size, num_workers=opt.num_workers, seed=opt.trial)
        
        model_path = './assets/aug_exp_model/'+'{}-{}-ADAM-1-{}.pt'.format('Aug_eff',opt.mname,opt.trial)
        model = load_teacher(opt.mname,sm_dict,model_path,1)
        if opt.multigpu:
            model =nn.DataParallel(model,device_ids=[0,1])
    end_data = time.perf_counter()
    
    model.eval()

    if torch.cuda.is_available():
        model = model.to('cuda:0')

    start_process = time.perf_counter()
    with torch.no_grad():
        temp_time = 0
        for batch in tqdm(test_loader):
            start_event = torch.cuda.Event(enable_timing = True)
            end_event = torch.cuda.Event(enable_timing = True)
            bimg = batch[0].to('cuda:0')
            start_event.record()
            _ = model(bimg)
            end_event.record()
            torch.cuda.synchronize()
            temp_time += start_event.elapsed_time(end_event)
   
    end_process = time.perf_counter()
    end = time.perf_counter() 
    total_time = end-start
    total_data = end_data - start_data
    total_process = end_process - start_process
    print("==================================================================")
    print("{}-{}-{} Total Batch Processing Time (Not Consider Memory Loading) {}:".format(opt.mname,opt.batch_size,opt.mtype,opt.num_workers), temp_time *1e-3)
    print("{}-{}-{} Time upto DataLoader {}:".format(opt.mname,opt.batch_size,opt.mtype,opt.num_workers), total_data)
    print("{}-{}-{} Total Batch Processing Time (Batch Processing Time + Memory Loading) {}:".format(opt.mname,opt.batch_size,opt.mtype,opt.num_workers), total_process)
    print("{}-{}-{} Total Processing Time {}".format(opt.mname,opt.batch_size,opt.mtype,opt.num_workers), total_time)
    print("{}-{}-{} Total Memory Loading Time {}".format(opt.mname,opt.batch_size,opt.mtype,opt.num_workers), total_time - temp_time*1e-3)
    print("==================================================================")
    
if __name__ == '__main__':
    main()
