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
 

class SoundOODInstance(Dataset):
    # Single Channel Dataset 10m x 10m resolution
    def __init__(self, path, region, transform=None,target_transform=None):

        self.transform=transform
        self.target_transform=target_transform
        
        self.file_object = h5py.File(path+'/test_region_{}.h5py'.format(region), 'r')
        
    def __len__(self):
        return self.file_object['test_label'].shape[0]

    def __getitem__(self, index):
        one_img = self.file_object['test_img10'][index].reshape(1,100,100)/255.0
        
        img=torch.tensor(one_img,dtype=torch.float)
        target=torch.tensor(self.file_object['test_label'][index],dtype=torch.float)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
                
        return img, target, index

class SoundOODTwoInstance(Dataset):
    # Multi + Expansion Channel Dataset 10m x 10m, 1m x 1m resolution
    def __init__(self, path, region, transform=None,target_transform=None):

        self.transform=transform
        self.target_transform=target_transform
        
        self.file_object = h5py.File(path+'/test_region_{}.h5py'.format(region), 'r')
            
    def __len__(self):
        return self.file_object['test_label'].shape[0]

    def __getitem__(self, index):
        one_img = self.file_object['test_img10'][index].reshape(1,100,100)/255.0
        expansion_img = self.file_object['test_img1'][index].reshape(1,100,100)/255.0

        origin_img=torch.tensor(one_img,dtype=torch.float)
        expansion_img=torch.tensor(expansion_img,dtype=torch.float)
        target=torch.tensor(self.file_object['test_label'][index],dtype=torch.float)

        img = torch.cat([expansion_img, origin_img],dim=0)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target, index



class SoundOODDetachedOriginInstance(Dataset):
    # Multi Channel Dataset 10m x 10m resolution
    def __init__(self, path, region, transform=None,target_transform=None):

        self.transform=transform
        self.target_transform=target_transform
        
        self.file_object = h5py.File(path+'/test_region_{}.h5py'.format(region), 'r')
        
    def __len__(self):
        return self.file_object['test_label'].shape[0]

    def __getitem__(self, index):
        one_img = self.file_object['test_img10'][index].reshape(1,100,100)/255.0
        multi_channel_img = np.concatenate([np.ones((2,100,100)),one_img],axis=0)

        multi_channel_img[0,:,:] = np.where(multi_channel_img[2]<=90/255.0, multi_channel_img[2],1.0)
        multi_channel_img[1,:,:] = np.where(multi_channel_img[2]>90/255.0, multi_channel_img[2],1.0)
        
        img=torch.tensor(multi_channel_img,dtype=torch.float)
        target=torch.tensor(self.file_object['test_label'][index],dtype=torch.float)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
                
        return img, target, index


class SoundOODDetachedInstance(Dataset):
    # Multi + Expansion Channel Dataset 10m x 10m, 1m x 1m resolution
    def __init__(self, path, region, transform=None,target_transform=None):

        self.transform=transform
        self.target_transform=target_transform
        
        self.file_object = h5py.File(path+'/test_region_{}.h5py'.format(region), 'r')
            
    def __len__(self):
        return self.file_object['test_label'].shape[0]

    def __getitem__(self, index):
        one_img = self.file_object['test_img10'][index].reshape(1,100,100)/255.0
        expansion_img = self.file_object['test_img1'][index].reshape(1,100,100)/255.0
        multi_channel_img = np.concatenate([np.ones((2,100,100)),one_img],axis=0)

        multi_channel_img[0,:,:] = np.where(multi_channel_img[2]<=90/255.0, multi_channel_img[2],1.0)
        multi_channel_img[1,:,:] = np.where(multi_channel_img[2]>90/255.0, multi_channel_img[2],1.0)
        
        origin_img=torch.tensor(multi_channel_img,dtype=torch.float)
        expansion_img=torch.tensor(expansion_img,dtype=torch.float)
        target=torch.tensor(self.file_object['test_label'][index],dtype=torch.float)

        img = torch.cat([expansion_img, origin_img],dim=0)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target, index

    

def get_ood_detached_origin_sound_dataloaders(path,region, mean_vec, std_vec, batch_size=128, num_workers=4):
    
    """
    Sound Map multi channel data
    """
    
    test_transform = transforms.Compose([
        transforms.Normalize(mean_vec, std_vec),
    ])
    
    test_set = SoundOODDetachedOriginInstance(path,region = region, transform = test_transform)

    test_loader = DataLoader(test_set, batch_size=batch_size,shuffle=False,num_workers=num_workers)

    return test_loader

def get_ood_detached_sound_dataloaders(path,region, mean_vec, std_vec, batch_size=128, num_workers=4):
    
    """
    Sound Map multi + expansion data
    """
    test_transform = transforms.Compose([
        transforms.Normalize(mean_vec, std_vec),
    ])
    
    test_set = SoundOODDetachedInstance(path,region = region, transform = test_transform)

    test_loader = DataLoader(test_set, batch_size=batch_size,shuffle=False,num_workers=num_workers)

    return test_loader

def get_ood_sound_dataloaders(path,region, mean_vec, std_vec, batch_size=128, num_workers=4):
    
    """
    Sound Map multi + expansion data
    """
    test_transform = transforms.Compose([
        transforms.Normalize(mean_vec, std_vec),
    ])
    
    test_set = SoundOODInstance(path,region = region, transform = test_transform)

    test_loader = DataLoader(test_set, batch_size=batch_size,shuffle=False,num_workers=num_workers)

    return test_loader

def get_ood_two_sound_dataloaders(path,region, mean_vec, std_vec, batch_size=128, num_workers=4):
    
    """
    Sound Map multi + expansion data
    """
    test_transform = transforms.Compose([
        transforms.Normalize(mean_vec, std_vec),
    ])
    
    test_set = SoundOODTwoInstance(path,region = region, transform = test_transform)

    test_loader = DataLoader(test_set, batch_size=batch_size,shuffle=False,num_workers=num_workers)

    return test_loader