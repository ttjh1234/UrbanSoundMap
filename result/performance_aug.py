import torch
import numpy as np
import pandas as pd
from utils.util import *
from tqdm import tqdm
from models import model_dict as sm_dict
from dataset.sound_exp import get_sound_valid_dataloaders
import random
import argparse

def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--epochs', type=int, default=40, help='number of training epochs')
    parser.add_argument('--device', type=int, default=0, help='GPU number')
    parser.add_argument('--num_workers', type=int, default=0, help='Num Worker')
    parser.add_argument('--multigpu', type=bool, default=False, help='multigpu')

    parser.add_argument('--atype',type=str, default='total')
    parser.add_argument('--method',type=str, default='no_aug', choices= ['No_aug','Aug','Aug_eff'])

    # Experiment
    #parser.add_argument('--trial', type=int, default=0, help='the experiment id')

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
    model_name_list = ['vgg11','vgg13','vgg16','vgg19','resnet18','wrn_16_2','wrn_40_2','inception_resnetv2_2']
    
    result_dict = {}
    for mname in model_name_list:
        result_dict[mname] = []

    
    for i in range(10):
        set_random_seed(i)

        _, val_loader, n_data= get_sound_valid_dataloaders(path='./assets/newdata/',batch_size=opt.batch_size, num_workers=0,seed=i,atype=opt.atype)
        n_cls = 1
        mdict = sm_dict

        for mname in model_name_list:

            model_path = './assets/aug_exp_model/'+'{}-{}-ADAM-1-{}.pt'.format(opt.method,mname,i)
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

            result_dict[mname].append(rmse_metric(pred,target))
    
    
    result_csv = pd.DataFrame(result_dict).T
    mean_vec = result_csv.mean(axis=1)
    std_vec = result_csv.std(axis=1)
    result_csv['mean'] = mean_vec
    result_csv['std'] = std_vec
    
    result_csv.to_csv('./assets/paper_result_file/augmentation_{}_{}.csv'.format(opt.atype,opt.method.lower()))
    
if __name__=="__main__":
    main()
    

