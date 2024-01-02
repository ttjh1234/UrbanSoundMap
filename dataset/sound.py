import torch
import numpy as np
import os
import sys
from collections import defaultdict as ddict
from torch.utils.data import TensorDataset, DataLoader,Dataset, random_split
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import hflip
from PIL import Image
from scipy import interpolate

class SoundInstance(Dataset):
    def __init__(self, path, train=True, transform=None,target_transform=None):

        self.transform=transform
        self.target_transform=target_transform
        self.train=train
               
        if self.train:
            self.img=np.load(path+'/train_img.npy')
            self.label=np.load(path+'/train_label.npy')
        
        else:
            self.img=np.load(path+'/test_img.npy')
            self.label=np.load(path+'/test_label.npy')
            
    def __len__(self):
        return self.img.shape[0]

    def __getitem__(self, index):

        img=torch.tensor(self.img[index],dtype=torch.float).unsqueeze(0)
        target=torch.tensor(self.label[index],dtype=torch.float)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

def get_sound_dataloaders(path,batch_size=128, num_workers=4):
    
    """
    Sound Map data
    """
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize((0.9113), (0.2168)),
    ])
    
    test_transform = transforms.Compose([
        transforms.Normalize((0.9113), (0.2168)),
    ])


    train_set = SoundInstance(path,train = True, transform = train_transform)
    n_data = len(train_set)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = SoundInstance(path, train=False, transform = test_transform)

    test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=num_workers)


    return train_loader, test_loader, n_data
