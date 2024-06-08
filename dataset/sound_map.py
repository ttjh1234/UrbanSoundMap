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
 
class SoundMapInstance(Dataset):
    # Single Channel Dataset 10m x 10m resolution
    def __init__(self, path, t_mean=None, t_std = None,transform=None,target_transform=None):

        self.transform=transform
        self.target_transform=target_transform

        self.train_img10=np.load(path+'/train_img10.npy')
        self.train_img1 = np.load(path+'/train_img1.npy')
        self.train_label=np.load(path+'/train_label.npy')
    
        self.test_img10=np.load(path+'/test_img10.npy')
        self.test_img1=np.load(path+'/test_img1.npy')
        self.test_label=np.load(path+'/test_label.npy')
        
        self.total_img10 = np.concatenate([self.train_img10,self.test_img10],axis=0)
        self.total_img1 = np.concatenate([self.train_img1,self.test_img1],axis=0)

        del self.train_img10
        del self.train_img1
        del self.test_img10
        del self.test_img1        

        self.total_label = np.concatenate([self.train_label,self.test_label],axis=0)

        if (t_mean is None) & (t_std is None):
            self.mean_value, self.std_value = self.get_mean_var()
        else:
            self.mean_value = t_mean
            self.std_value = t_std
        
    def get_mean_var(self):
        # 3 channel         
        mean_value0 = np.mean(np.mean(self.total_img1,axis=(1,2)))
        std_value0 = np.mean(np.std(self.total_img1,axis=(1,2)))
        
        mean_value1 = np.mean(np.mean(self.total_img10,axis=(1,2)))
        std_value1 = np.mean(np.std(self.total_img10,axis=(1,2)))

        mean_vec = torch.tensor((mean_value0,mean_value1),dtype=torch.float).unsqueeze(1).unsqueeze(1)
        std_vec = torch.tensor((std_value0,std_value1),dtype=torch.float).unsqueeze(1).unsqueeze(1)

        return mean_vec, std_vec

    def normalizing(self, x):
        return (x - self.mean_value)/self.std_value
        
    def __len__(self):
        return self.total_label.shape[0]

    def __getitem__(self, index):

        img10=torch.tensor(self.total_img10[index],dtype=torch.float).unsqueeze(0)
        img1=torch.tensor(self.total_img1[index],dtype=torch.float).unsqueeze(0)
        target=torch.tensor(self.total_label[index],dtype=torch.float)

        img = torch.cat([img1,img10],dim=0)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        img = self.normalizing(img)

        return img, target, index


class SoundMapSingleInstance(Dataset):
    # Single Channel Dataset 10m x 10m resolution
    def __init__(self, path, t_mean=None, t_std = None,transform=None,target_transform=None):

        self.transform=transform
        self.target_transform=target_transform

        self.train_img10=np.load(path+'/train_img10.npy')
        self.train_label=np.load(path+'/train_label.npy')
    
        self.test_img10=np.load(path+'/test_img10.npy')
        self.test_label=np.load(path+'/test_label.npy')
        
        self.total_img10 = np.concatenate([self.train_img10,self.test_img10],axis=0)

        del self.train_img10
        del self.test_img10

        self.total_label = np.concatenate([self.train_label,self.test_label],axis=0)

        if (t_mean is None) & (t_std is None):
            self.mean_value, self.std_value = self.get_mean_var()
        else:
            self.mean_value = t_mean
            self.std_value = t_std
        
    def get_mean_var(self):
        # 3 channel         
        
        mean_value0 = np.mean(np.mean(self.total_img10,axis=(1,2)),keepdims=True)
        std_value0 = np.mean(np.std(self.total_img10,axis=(1,2)),keepdims=True)

        mean_vec = torch.tensor(mean_value0,dtype=torch.float).unsqueeze(1).unsqueeze(1)
        std_vec = torch.tensor(std_value0,dtype=torch.float).unsqueeze(1).unsqueeze(1)

        return mean_vec, std_vec

    def normalizing(self, x):
        return (x - self.mean_value)/self.std_value
        
    def __len__(self):
        return self.total_label.shape[0]

    def __getitem__(self, index):

        img=torch.tensor(self.total_img10[index],dtype=torch.float).unsqueeze(0)
        target=torch.tensor(self.total_label[index],dtype=torch.float)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        img = self.normalizing(img)

        return img, target, index

def get_soundmap_dataloaders(path,batch_size=128, num_workers=4):
    
    """
    Sound Map data efficient aug Experiment for single channel dataset
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        #transforms.Normalize((0.90865), (0.21802)),
    ])
    
    # test_transform = transforms.Compose([
    #     transforms.Normalize((0.90865), (0.21802)),
    # ])


    train_set = SoundMapInstance(path, transform = train_transform)
    n_data = len(train_set)
    
    t_mean = train_set.mean_value
    t_std = train_set.std_value

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    return train_loader, t_mean,t_std, n_data

def get_single_soundmap_dataloaders(path,batch_size=128, num_workers=4):
    
    """
    Sound Map data efficient aug Experiment for single channel dataset
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        #transforms.Normalize((0.90865), (0.21802)),
    ])
    
    # test_transform = transforms.Compose([
    #     transforms.Normalize((0.90865), (0.21802)),
    # ])


    train_set = SoundMapSingleInstance(path, transform = train_transform)
    n_data = len(train_set)
    
    t_mean = train_set.mean_value
    t_std = train_set.std_value

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    return train_loader, t_mean,t_std, n_data
