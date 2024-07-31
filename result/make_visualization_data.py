import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import model_dict as sm_dict
from two_models import model_dict as tm_dict
from utils.util import set_random_seed, load_teacher
from dataset.sound import get_mean_std_two,get_mean_std_single,get_sound_two_total_dataloaders,get_sound_total_dataloaders

def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--device', type=int, default=0, help='GPU number')
    parser.add_argument('--num_workers', type=int, default=0, help='Num Worker')
    parser.add_argument('--multigpu', type=bool, default=False, help='multigpu')
    
    # dataset
    parser.add_argument('--model', type=str, default='vgg13',
                        choices=['vgg8','vgg11','vgg13','vgg16','vgg19','resnet18','wrn_16_2','wrn_40_2'])
    
    parser.add_argument('--mtype', type=str, default='single',choices=['single','two'])
    
    opt = parser.parse_args()
        
    if opt.device==0:
        opt.device='cuda:0'
    elif opt.device==1:
        opt.device='cuda:1'
    else:
        opt.device='cpu'

    return opt

def main():

    opt = parse_option()
    
    set_random_seed(0)
    path = './assets/newdata/'
    # dataloader
    if opt.mtype == 'single':
        
        mean_vec, std_vec = get_mean_std_single(path=path,seed=0)
        total_loader = get_sound_total_dataloaders(path, batch_size=opt.batch_size, mean_vec =mean_vec, std_vec = std_vec, num_workers = opt.num_workers)                
        n_cls = 1
        model_dict = sm_dict

        model_path = './assets/aug_exp_model/'+'{}-{}-ADAM-1-{}.pt'.format('Aug_eff',opt.model,0)
        model = load_teacher(opt.model,model_dict,model_path,n_cls)
        # model
        if opt.multigpu:
            model =nn.DataParallel(model,device_ids=[0,1])
        
    elif opt.mtype == 'two':
        
        mean_vec, std_vec = get_mean_std_two(path=path,seed=0)
        total_loader = get_sound_two_total_dataloaders(path, batch_size=opt.batch_size, mean_vec =mean_vec, std_vec = std_vec, num_workers = opt.num_workers)
        n_cls = 1
        model_dict = tm_dict

        # model        
        model_path = './assets/model/'+'{}-{}-ADAM-1-{}.pt'.format('two',opt.model,0)
        model = load_teacher(opt.model,model_dict,model_path,n_cls)
        if opt.multigpu:
            model =nn.DataParallel(model,device_ids=[0,1])

        
    else:
        raise NotImplementedError()


 
    if torch.cuda.is_available():
        model = model.to(opt.device)

    prediction = np.zeros((0,))
    target = np.zeros((0,))
    coord = np.zeros((0,2))
    model.eval()

    with torch.no_grad():
        for _, batch in enumerate(total_loader):

            temp_src = batch[0].to(opt.device)
            temp_trg = batch[1].numpy()
            temp_coord = batch[2].numpy()
            output = model(temp_src)
            output = output.squeeze(1).detach().to('cpu').numpy()
            
            prediction = np.concatenate([prediction, output],axis=0)
            target = np.concatenate([target, temp_trg],axis=0)
            coord = np.concatenate([coord, temp_coord],axis=0)
            
        
    np.save('./assets/vdata/total_pred_{}_{}_0'.format(opt.mtype, opt.model),prediction)
    np.save('./assets/vdata/total_label_{}_{}_0'.format(opt.mtype, opt.model),target)   
    np.save('./assets/vdata/total_coord_{}_{}_0'.format(opt.mtype, opt.model),coord)

if __name__ == '__main__':
    main()
