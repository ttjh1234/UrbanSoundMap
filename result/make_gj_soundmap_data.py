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
from two_models import model_dict
from utils.util import load_teacher
from dataset.sound_hdf5 import get_total_dataloaders

def main():
    test_loader, num_data = get_total_dataloaders(path = './assets/', batch_size= 128, num_workers=0, seed=0)
    
    mname = 'vgg13'
    model_path = './assets/sound_map/'+'{}-{}-ADAM-1-{}.pt'.format('two',mname,0)
    model = load_teacher(mname,model_dict,model_path,1)
    
    model.eval()

    if torch.cuda.is_available():
        model = model.to('cuda:0')

    print(num_data)
    tot_pred = np.zeros((0,))
    tot_label = np.zeros((0,))
    tot_coord = np.zeros((0,2))

    with torch.no_grad():
        for batch in tqdm(test_loader):
            temp_img = batch[0].to('cuda:0')
            tot_label = np.concatenate([tot_label, batch[1].numpy()],axis=0)
            tot_coord = np.concatenate([tot_coord, batch[2].numpy()],axis=0)
            
            temp_pred = model(temp_img).squeeze(1).to('cpu').numpy()
            tot_pred = np.concatenate([tot_pred, temp_pred])
    
    np.save('./assets/sound_map/tot_pred', tot_pred)        
    np.save('./assets/sound_map/tot_label', tot_label)        
    np.save('./assets/sound_map/tot_coord', tot_coord)        

if __name__ == '__main__':
    main()
