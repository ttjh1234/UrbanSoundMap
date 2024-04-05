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


def srs_group(total_size, sample_size):
    sample_idx = np.random.choice(range(total_size),size=sample_size, replace=False)
    sample_idx = np.sort(sample_idx)
     
    return sample_idx
    

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
 

class SoundSampleOODInstance(Dataset):
    # Single Channel Dataset 10m x 10m resolution
    def __init__(self, path, region, sample_size, transform=None,target_transform=None):

        self.transform=transform
        self.target_transform=target_transform
        
        self.file_object = h5py.File(path+'/test_region_{}.h5py'.format(region), 'r')

        sample_idx = srs_group(self.file_object['test_label'].shape[0], sample_size)
        self.img = self.file_object['test_img10'][sample_idx]
        self.label = self.file_object['test_label'][sample_idx]
    
        
    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, index):
        one_img = self.img[index].reshape(1,100,100)/255.0
        
        img=torch.tensor(one_img,dtype=torch.float)
        target=torch.tensor(self.label[index],dtype=torch.float)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
                
        return img, target, index

class SoundSampleOODTwoInstance(Dataset):
    # Multi + Expansion Channel Dataset 10m x 10m, 1m x 1m resolution
    def __init__(self, path, region, sample_size, transform=None,target_transform=None):

        self.transform=transform
        self.target_transform=target_transform
        
        self.file_object = h5py.File(path+'/test_region_{}.h5py'.format(region), 'r')
        sample_idx = srs_group(self.file_object['test_label'].shape[0], sample_size)

        self.img10 = self.file_object['test_img10'][sample_idx]
        self.img1 = self.file_object['test_img1'][sample_idx]
        self.label = self.file_object['test_label'][sample_idx]
         
    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, index):
        one_img = self.img10[index].reshape(1,100,100)/255.0
        expansion_img = self.img1[index].reshape(1,100,100)/255.0

        origin_img=torch.tensor(one_img,dtype=torch.float)
        expansion_img=torch.tensor(expansion_img,dtype=torch.float)
        target=torch.tensor(self.label[index],dtype=torch.float)

        img = torch.cat([expansion_img, origin_img],dim=0)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target, index



class SoundSampleOODDetachedOriginInstance(Dataset):
    # Multi Channel Dataset 10m x 10m resolution
    def __init__(self, path, region, sample_size, transform=None,target_transform=None):

        self.transform=transform
        self.target_transform=target_transform
        
        self.file_object = h5py.File(path+'/test_region_{}.h5py'.format(region), 'r')
        sample_idx = srs_group(self.file_object['test_label'].shape[0], sample_size)

        self.img10 = self.file_object['test_img10'][sample_idx]
        self.label = self.file_object['test_label'][sample_idx]
        
        
    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, index):
        one_img = self.img10[index].reshape(1,100,100)/255.0
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


class SoundSampleOODDetachedInstance(Dataset):
    # Multi + Expansion Channel Dataset 10m x 10m, 1m x 1m resolution
    def __init__(self, path, region, sample_size, transform=None,target_transform=None):

        self.transform=transform
        self.target_transform=target_transform
        
        self.file_object = h5py.File(path+'/test_region_{}.h5py'.format(region), 'r')
        sample_idx = srs_group(self.file_object['test_label'].shape[0], sample_size)
        self.img10 = self.file_object['test_img10'][sample_idx]
        self.img1 = self.file_object['test_img1'][sample_idx]
        self.label = self.file_object['test_label'][sample_idx]
        
    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, index):
        one_img = self.img10[index].reshape(1,100,100)/255.0
        expansion_img = self.img1[index].reshape(1,100,100)/255.0
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

    

def get_sample_ood_detached_origin_sound_dataloaders(path,region, sample_size, mean_vec, std_vec, batch_size=128, num_workers=4):
    
    """
    Sound Map multi channel data
    """
    
    test_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize(mean_vec, std_vec),
    ])
    
    test_set = SoundSampleOODDetachedOriginInstance(path,region = region,sample_size=sample_size, transform = test_transform)

    test_loader = DataLoader(test_set, batch_size=batch_size,shuffle=True,num_workers=num_workers)

    return test_loader

def get_sample_ood_detached_sound_dataloaders(path,region, sample_size, mean_vec, std_vec, batch_size=128, num_workers=4):
    
    """
    Sound Map multi + expansion data
    """
    test_transform = transforms.Compose([
        SoundRandomHorizontalFlip(),
        SoundRandomVerticalFlip(),
        transforms.Normalize(mean_vec, std_vec),
    ])
    
    test_set = SoundSampleOODDetachedInstance(path,region = region,sample_size=sample_size, transform = test_transform)

    test_loader = DataLoader(test_set, batch_size=batch_size,shuffle=True,num_workers=num_workers)

    return test_loader

def get_sample_ood_sound_dataloaders(path,region,sample_size, mean_vec, std_vec, batch_size=128, num_workers=4):
    
    """
    Sound Map multi + expansion data
    """
    test_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize(mean_vec, std_vec),
    ])
    
    test_set = SoundSampleOODInstance(path,region = region,sample_size=sample_size, transform = test_transform)

    test_loader = DataLoader(test_set, batch_size=batch_size,shuffle=True,num_workers=num_workers)

    return test_loader

def get_sample_ood_two_sound_dataloaders(path,region,sample_size, mean_vec, std_vec, batch_size=128, num_workers=4):
    
    """
    Sound Map multi + expansion data
    """
    test_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize(mean_vec, std_vec),
    ])
    
    test_set = SoundSampleOODTwoInstance(path,region = region,sample_size=sample_size, transform = test_transform)

    test_loader = DataLoader(test_set, batch_size=batch_size,shuffle=True,num_workers=num_workers)

    return test_loader