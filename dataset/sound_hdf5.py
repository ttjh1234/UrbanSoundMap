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
from torchvision.transforms.functional import hflip, vflip
from PIL import Image
from scipy import interpolate
import random
import pandas as pd
import h5py

# Sound Augmentation Random Horizontal Flip and Random Vertical Flip
class SoundRandomHorizontalFlip(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()

        assert (p>=0) & (p<=1)
        self.p=p
    
    def forward(self, data):
        if torch.rand(1)<self.p:
            img=data[1:,:,:]
            expansion=data[:1,:,:]
            flip_img=hflip(img)
            flip_expansion=hflip(expansion)
            data=torch.cat([flip_expansion, flip_img],dim=0)

        return data
    
class SoundRandomVerticalFlip(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()

        assert (p>=0) & (p<=1)
        self.p=p
    
    def forward(self, data):
        if torch.rand(1)<self.p:
            img=data[1:,:,:]
            expansion=data[:1,:,:]
            flip_img=vflip(img)
            flip_expansion=vflip(expansion)
            data=torch.cat([flip_expansion, flip_img],dim=0)

        return data
 

class SoundDetachedOriginInstance(Dataset):
    # Multi Channel Dataset 10m x 10m resolution
    def __init__(self, path, seed, train=True, transform=None,target_transform=None):

        self.transform=transform
        self.target_transform=target_transform
        
        self.file_object = h5py.File(path+'/total_data_{}.h5py'.format(seed), 'r')
        if train:
            self.origin_img = self.file_object['train_img10']
            self.label = self.file_object['train_label']
        else:
            self.origin_img = self.file_object['test_img10']
            self.label = self.file_object['test_label']
        
    def __len__(self):
        return len(self.origin_img)

    def __getitem__(self, index):
        one_img = self.origin_img[index].reshape(1,100,100)/255.0
        multi_channel_img = np.concatenate([np.ones((2,100,100)),one_img],axis=0)

        multi_channel_img[0,:,:] = np.where(multi_channel_img[2]<=90/255.0, multi_channel_img[2],1.0)
        multi_channel_img[1,:,:] = np.where(multi_channel_img[2]>90/255.0, multi_channel_img[2],1.0)
        
        img=torch.tensor(multi_channel_img,dtype=torch.float)
        target=torch.tensor(self.label[index],dtype=torch.float)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
                
        return img, target, index


class SoundDetachedInstance(Dataset):
    # Multi + Expansion Channel Dataset 10m x 10m, 1m x 1m resolution
    def __init__(self, path, seed, train=True, transform=None,target_transform=None):

        self.transform=transform
        self.target_transform=target_transform
        
        self.file_object = h5py.File(path+'/total_data_{}.h5py'.format(seed), 'r')
        if train:
            self.origin_img = self.file_object['train_img10']
            self.expansion_img = self.file_object['train_img1']
            self.label = self.file_object['train_label']
        else:
            self.origin_img = self.file_object['test_img10']
            self.expansion_img = self.file_object['test_img1']
            self.label = self.file_object['test_label']
            
    def __len__(self):
        return len(self.origin_img)

    def __getitem__(self, index):
        one_img = self.origin_img[index].reshape(1,100,100)/255.0
        expansion_img = self.expansion_img[index].reshape(1,100,100)/255.0
        multi_channel_img = np.concatenate([np.ones((2,100,100)),one_img],axis=0)

        multi_channel_img[0,:,:] = np.where(multi_channel_img[2]<=90/255.0, multi_channel_img[2],1.0)
        multi_channel_img[1,:,:] = np.where(multi_channel_img[2]>90/255.0, multi_channel_img[2],1.0)
        
        origin_img=torch.tensor(multi_channel_img,dtype=torch.float)
        expansion_img=torch.tensor(expansion_img,dtype=torch.float)
        target=torch.tensor(self.label[index],dtype=torch.float)

        img = torch.cat([expansion_img, origin_img],dim=0)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target, index

    

def get_detached_origin_sound_dataloaders(path,batch_size=128, num_workers=4, seed=0):
    
    """
    Sound Map multi channel data
    """
    
    summary_statistics = pd.read_csv(path+'total_data_summary_{}.csv'.format(seed))
    mean_vec = np.array(summary_statistics.iloc[0][['build','road','origin']])
    std_vec = np.array(summary_statistics.iloc[1][['build','road','origin']])
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize(mean_vec, std_vec),
    ])
    
    test_transform = transforms.Compose([
        transforms.Normalize(mean_vec, std_vec),
    ])


    train_set = SoundDetachedOriginInstance(path,seed=seed, train=True, transform = train_transform)
    n_data = len(train_set)
    
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = SoundDetachedOriginInstance(path,seed=seed, train=False, transform = test_transform)

    test_loader = DataLoader(test_set, batch_size=batch_size,shuffle=False,num_workers=num_workers)


    return train_loader, test_loader, n_data

def get_detached_sound_dataloaders(path,batch_size=128, num_workers=4, seed=0):
    
    """
    Sound Map multi + expansion data
    """
    summary_statistics = pd.read_csv(path+'total_data_summary_{}.csv'.format(seed))
    mean_vec = np.array(summary_statistics.iloc[0])
    std_vec = np.array(summary_statistics.iloc[1])
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize(mean_vec, std_vec),
    ])
    
    test_transform = transforms.Compose([
        transforms.Normalize(mean_vec, std_vec),
    ])

    train_set = SoundDetachedInstance(path,seed=seed, transform = train_transform)
    n_data = len(train_set)
    
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = SoundDetachedInstance(path,seed=seed, train=False, transform = test_transform)

    test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=num_workers)


    return train_loader, test_loader, n_data


