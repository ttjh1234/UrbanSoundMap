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
        
        self.file_object = h5py.File(path+'/total_data/total_data_{}.h5py'.format(seed), 'r')
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


class SoundSingleTotal(Dataset):
    # Multi + Expansion Channel Dataset 10m x 10m, 1m x 1m resolution
    def __init__(self, path, seed, transform=None,target_transform=None):

        self.transform=transform
        self.target_transform=target_transform
        
        self.file_object = h5py.File(path+'/total_data/total_data_{}.h5py'.format(seed), 'r')

        self.t_origin_img = self.file_object['train_img10']
        self.t_label = self.file_object['train_label']
        self.t_coord = self.file_object['train_coord']

        self.v_origin_img = self.file_object['test_img10']
        self.v_label = self.file_object['test_label']
        self.v_coord = self.file_object['test_coord']
        
        self.origin_img = np.concatenate([self.t_origin_img, self.v_origin_img],axis=0)
        del self.t_origin_img
        del self.v_origin_img
        
        
        self.label = np.concatenate([self.t_label, self.v_label],axis=0)
        self.coord = np.concatenate([self.t_coord, self.v_coord],axis=0)
        del self.t_label
        del self.v_label
        del self.t_coord
        del self.v_coord
        
    def __len__(self):
        return len(self.origin_img)
    

    def __getitem__(self, index):
        one_img = self.origin_img[index].reshape(1,100,100)/255.0
                
        origin_img=torch.tensor(one_img,dtype=torch.float)
        target=torch.tensor(self.label[index],dtype=torch.float)
        coord=torch.tensor(self.coord[index],dtype=torch.long)

        img = origin_img

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target, coord, index

class SoundTotal(Dataset):
    # Multi + Expansion Channel Dataset 10m x 10m, 1m x 1m resolution
    def __init__(self, path, seed, transform=None,target_transform=None):

        self.transform=transform
        self.target_transform=target_transform
        
        self.file_object = h5py.File(path+'/total_data/total_data_{}.h5py'.format(seed), 'r')

        self.t_origin_img = self.file_object['train_img10']
        self.t_expansion_img = self.file_object['train_img1']
        self.t_label = self.file_object['train_label']
        self.t_coord = self.file_object['train_coord']

        self.v_origin_img = self.file_object['test_img10']
        self.v_expansion_img = self.file_object['test_img1']
        self.v_label = self.file_object['test_label']
        self.v_coord = self.file_object['test_coord']
        
        self.origin_img = np.concatenate([self.t_origin_img, self.v_origin_img],axis=0)
        del self.t_origin_img
        del self.v_origin_img
        
        self.expansion_img = np.concatenate([self.t_expansion_img, self.v_expansion_img],axis=0)
        del self.t_expansion_img
        del self.v_expansion_img
        
        self.label = np.concatenate([self.t_label, self.v_label],axis=0)
        self.coord = np.concatenate([self.t_coord, self.v_coord],axis=0)
        del self.t_label
        del self.v_label
        del self.t_coord
        del self.v_coord
        
    def __len__(self):
        return len(self.origin_img)

    def __getitem__(self, index):
        one_img = self.origin_img[index].reshape(1,100,100)/255.0
        expansion_img = self.expansion_img[index].reshape(1,100,100)/255.0
                
        origin_img=torch.tensor(one_img,dtype=torch.float)
        expansion_img=torch.tensor(expansion_img,dtype=torch.float)
        target=torch.tensor(self.label[index],dtype=torch.float)
        coord=torch.tensor(self.coord[index],dtype=torch.long)

        img = torch.cat([expansion_img, origin_img],dim=0)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target, coord, index


class SoundMLTotal(Dataset):
    # Multi + Expansion Channel Dataset 10m x 10m, 1m x 1m resolution
    def __init__(self, path, mean_v, std_v):

        self.file_object = h5py.File(path+'/newdata/urbanform1000pool2_total.h5py', 'r')
        self.data = self.file_object['data']
        #self.data = np.array(self.file_object['data'])
        self.mean_value = mean_v
        self.std_value = std_v
        
    def __len__(self):
        return self.data.shape[0]
    
    def normalizing(self, x):
        return (x - self.mean_value)/self.std_value

    def __getitem__(self, index):
        data = self.data[index].astype(float)
        data = self.normalizing(data)
        
        return data

class SoundSingleMLTotal(Dataset):
    # Multi + Expansion Channel Dataset 10m x 10m, 1m x 1m resolution
    def __init__(self, path, mean_v, std_v):
 
        self.file_object = h5py.File(path+'/total_data/total_data_0.h5py', 'r')

        self.t_origin_img = self.file_object['train_img10']
        self.v_origin_img = self.file_object['test_img10']

        
        self.origin_img = np.concatenate([self.t_origin_img, self.v_origin_img],axis=0)
        del self.t_origin_img
        del self.v_origin_img
        
        self.mean_value = mean_v
        self.std_value = std_v
        
    def __len__(self):
        return len(self.origin_img)
    
    def normalizing(self, x):
        return (x - self.mean_value)/self.std_value

    def __getitem__(self, index):
        # 70000
        data= self.origin_img[index].reshape(-1)/255.0
        data = self.normalizing(data)
        
        return data

class SoundTwoMLTotal(Dataset):
    # Multi + Expansion Channel Dataset 10m x 10m, 1m x 1m resolution
    def __init__(self, path, mean_v, std_v):

        self.file_object = h5py.File(path+'/total_data/total_data_0.h5py', 'r')

        self.t_origin_img = self.file_object['train_img10']
        self.t_expansion_img = self.file_object['train_img1']
        
        self.v_origin_img = self.file_object['test_img10']
        self.v_expansion_img = self.file_object['test_img1']
        
        
        self.origin_img = np.concatenate([self.t_origin_img, self.v_origin_img],axis=0)
        del self.t_origin_img
        del self.v_origin_img
        
        self.expansion_img = np.concatenate([self.t_expansion_img, self.v_expansion_img],axis=0)
        del self.t_expansion_img
        del self.v_expansion_img
        
        self.mean_value = mean_v
        self.std_value = std_v
        
    def __len__(self):
        return len(self.origin_img)

    def normalizing(self, x):
        return (x - self.mean_value)/self.std_value

    def __getitem__(self, index):
        origin_data = self.origin_img[index].reshape(-1)/255.0
        expansion_data = self.expansion_img[index].reshape(-1)/255.0
                        
        data = np.concatenate([expansion_data, origin_data],axis=0)
        data = self.normalizing(data)
        
        return data


def get_ml_total_dataloaders(path,batch_size=128, num_workers=4, seed=0):
    mean_v = np.load(path+'/newdata/ml_mean.npy')
    std_v = np.load(path+'/newdata/ml_std.npy')

    test_set = SoundMLTotal(path, mean_v, std_v)
    num_data = len(test_set)
    test_loader = DataLoader(test_set, batch_size=batch_size,shuffle=False,num_workers=num_workers)

    return test_loader, num_data

def get_ml_sv_total_dataloaders(path,batch_size=128, num_workers=4, seed=0):
    mean_v = np.load(path+'/newdata/ml_single_mean.npy')
    std_v = np.load(path+'/newdata/ml_single_std.npy')

    test_set = SoundSingleMLTotal(path, mean_v, std_v)
    num_data = len(test_set)
    test_loader = DataLoader(test_set, batch_size=batch_size,shuffle=False,num_workers=num_workers)

    return test_loader, num_data

def get_ml_svp_total_dataloaders(path,batch_size=128, num_workers=4, seed=0):
    mean_v = np.load(path+'/newdata/ml_two_mean.npy')
    std_v = np.load(path+'/newdata/ml_two_std.npy')

    test_set = SoundTwoMLTotal(path, mean_v, std_v)
    num_data = len(test_set)
    test_loader = DataLoader(test_set, batch_size=batch_size,shuffle=False,num_workers=num_workers)

    return test_loader, num_data


def get_single_total_dataloaders(path,batch_size=128, num_workers=4, seed=0):
    mean_v = np.load(path+'/sound_map/mean_vec.npy')
    std_v = np.load(path+'/sound_map/std_vec.npy')
    test_transform = transforms.Compose([
        transforms.Normalize(mean_v[[-1]], std_v[[-1]]),
    ])

    test_set = SoundSingleTotal(path,seed=seed, transform = test_transform)
    num_data = len(test_set)
    test_loader = DataLoader(test_set, batch_size=batch_size,shuffle=False,num_workers=num_workers)

    return test_loader, num_data

def get_total_dataloaders(path,batch_size=128, num_workers=4, seed=0):
    mean_v = np.load(path+'/sound_map/mean_vec.npy')
    std_v = np.load(path+'/sound_map/std_vec.npy')
    test_transform = transforms.Compose([
        transforms.Normalize(mean_v, std_v),
    ])

    test_set = SoundTotal(path,seed=seed, transform = test_transform)
    num_data = len(test_set)
    test_loader = DataLoader(test_set, batch_size=batch_size,shuffle=False,num_workers=num_workers)

    return test_loader, num_data

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


