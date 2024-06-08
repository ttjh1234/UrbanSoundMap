import argparse
import time
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import math
from tqdm import tqdm

from two_models import model_dict as tm_dict
from dataset.sound_ood import get_ood_two_eval_sound_dataloaders
from dataset.sound_da2 import get_sample_ood_two_sound_dataloaders

def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--device', type=int, default=0, help='GPU number')
    parser.add_argument('--num_workers', type=int, default=0, help='Num Worker')
    
    # dataset
    parser.add_argument('--method', type=str, default='single',
                        choices=['single','two','multi','four'], help = 'Channel Architecture')    
    parser.add_argument('--model', type=str, default='vgg13',
                        choices=['vgg8','vgg11','vgg13','vgg16','vgg19','resnet18','wrn_16_2','wrn_40_2'])
    parser.add_argument('--dataset', type=str, default='DJ', choices=['DJ','Seoul'], help='dataset')
    parser.add_argument('--d_num', type=int, default=10, help = 'The number of Training Number')
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
        if opt.d_num !=0:
            _ , mean_v, std_v = get_sample_ood_two_sound_dataloaders(path='./assets/newdata/', region = opt.dataset, sample_size=opt.d_num,
                                                                            batch_size=opt.batch_size, num_workers=opt.num_workers)

            valid_loader = get_ood_two_eval_sound_dataloaders(path='./assets/newdata/', region = opt.dataset, mean_vec=mean_v, std_vec=std_v,
                                                                batch_size=512, num_workers=opt.num_workers)
            n_cls = 1
            mdict = tm_dict

            file_path = './assets/sound_map/dj_two_vgg13_da_{}.pt'.format(opt.d_num)
        else:
            # No Domain Adaptation
            mean_v = np.load('./assets/sound_map/mean_vec.npy')
            std_v = np.load('./assets/sound_map/std_vec.npy')
            valid_loader = get_ood_two_eval_sound_dataloaders(path='./assets/newdata/', region = opt.dataset, mean_vec=mean_v, std_vec=std_v,
                                                                batch_size=512, num_workers=opt.num_workers)
            n_cls = 1
            mdict = tm_dict

            file_path = './assets/sound_map/'+'{}-{}-ADAM-1-{}.pt'.format('two',opt.model,opt.trial)
    else:
        raise NotImplementedError(opt.dataset)

    # model
    model = load_teacher(opt.model,mdict,file_path,n_cls)
    
    model.eval()
    if torch.cuda.is_available():
        model = model.to(opt.device)
        
    # routine - evaluate total_data
    tot_pred = np.zeros((0,))
    tot_label = np.zeros((0,))
    tot_coord = np.zeros((0,2))
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            temp_img = batch[0].to(opt.device)
            tot_label = np.concatenate([tot_label, batch[1].numpy()],axis=0)
            tot_coord = np.concatenate([tot_coord, batch[2].numpy()],axis=0)
            
            temp_pred = model(temp_img).squeeze(1).to('cpu').numpy()
            tot_pred = np.concatenate([tot_pred, temp_pred])
        
    np.save('./assets/sound_map_dj/dj_tot_pred_{}'.format(opt.d_num), tot_pred)        
    np.save('./assets/sound_map_dj/dj_tot_label_{}'.format(opt.d_num), tot_label)        
    np.save('./assets/sound_map_dj/dj_tot_coord_{}'.format(opt.d_num), tot_coord)            



if __name__ == '__main__':
    main()