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

region_dict= {}

region_dict[0] = [[740,1895],[1285,2351]]
region_dict[1] = [[1285,1895],[1830,2351]]
region_dict[2] = [[1830,1895],[2375,2351]]
region_dict[3] = [[2375,1895],[2920,2351]]

region_dict[4] = [[740,1439],[1285,1895]]
region_dict[5] = [[1285,1439],[1830,1895]]
region_dict[6] = [[1830,1439],[2375,1895]]
region_dict[7] = [[2375,1439],[2920,1895]]

region_dict[8] = [[195,982],[740,1439]]
region_dict[9] = [[740,982],[1285,1439]]
region_dict[10] = [[1285,982],[1830,1439]]
region_dict[11] = [[1830,982],[2375,1439]]
region_dict[12] = [[2375,982],[2920,1439]]

region_dict[13] = [[195,526],[740,982]]
region_dict[14] = [[740,526],[1285,982]]
region_dict[15] = [[1285,526],[1830,982]]
region_dict[16] = [[1830,526],[2375,982]]
region_dict[17] = [[2375,526],[2920,982]]

region_dict[18] = [[740,70],[1285,526]]
region_dict[19] = [[1285,70],[1830,526]]
region_dict[20] = [[1830,70],[2375,526]]
region_dict[21] = [[2375,70],[2920,526]]

def choose_region(seed):
    index_np = np.arange(0,22,1)
    valid_index = np.random.choice(index_np, 7, replace=False)
    train_index = np.delete(index_np,valid_index, axis=0)
    train_index = np.sort(train_index)
    valid_index = np.sort(valid_index)
    
    print("Seed : {}".format(seed))
    print("Train : ", train_index)
    print("Valid : ", valid_index)
             
    return train_index, valid_index

def parsing_index(img10, label, coord, index, img1=None):
    data_index = np.zeros((0,)).astype(int)
    for i in range(index.shape[0]):
        min_coord, max_coord = region_dict[index[i]]
        min_x, min_y = min_coord
        max_x, max_y = max_coord
        if max_y == 2351:
            if max_x == 2920:
                region_mask = ((coord[:,0]>=min_x) & (coord[:,0]<=max_x)) & ((coord[:,1]>=min_y) & (coord[:,1]<=max_y))
            else:
                region_mask = ((coord[:,0]>=min_x) & (coord[:,0]<max_x)) & ((coord[:,1]>=min_y) & (coord[:,1]<=max_y))
        else:
            if max_x == 2920:
                region_mask = ((coord[:,0]>=min_x) & (coord[:,0]<=max_x)) & ((coord[:,1]>=min_y) & (coord[:,1]<max_y))
            else:
                region_mask = ((coord[:,0]>=min_x) & (coord[:,0]<max_x)) & ((coord[:,1]>=min_y) & (coord[:,1]<max_y))
                
        data_index = np.concatenate([data_index, np.where(region_mask)[0]],axis=0)
    if img1 is not None:
        img10 = img10[data_index]
        img1 = img1[data_index]
        label = label[data_index]
        coord = coord[data_index]
        return img1, img10, label, coord
    else:
        img10 = img10[data_index]
        label = label[data_index]
        coord = coord[data_index]
        return img10, label, coord
    
    
class SoundMultiRandomHorizontalFlip(nn.Module):
    def __init__(self, p=0.5, mtype = 'single'):
        super().__init__()

        assert (p>=0) & (p<=1)
        self.p=p
        self.mtype = mtype

    def forward(self, x):
        if torch.rand(1)<self.p:
            data = x[0]
            label = x[1]
            if self.mtype == 'single':
                data=hflip(data)
                label = hflip(label)
            else:
                img=data[1:,:,:]
                expansion=data[:1,:,:]
                flip_img=hflip(img)
                flip_expansion=hflip(expansion)
                label = hflip(label)
                data=torch.cat([flip_expansion, flip_img],dim=0)
            x = (data,label)
        return x
    
class SoundMultiRandomVerticalFlip(nn.Module):
    def __init__(self, p=0.5, mtype = 'single'):
        super().__init__()

        assert (p>=0) & (p<=1)
        self.p=p
        self.mtype = mtype
        
    def forward(self, x):
        if torch.rand(1)<self.p:
            data = x[0]
            label = x[1]
            if self.mtype == 'single':
                data=vflip(data)
                label = vflip(label)
            else:
                img=data[1:,:,:]
                expansion=data[:1,:,:]
                flip_img=vflip(img)
                flip_expansion=vflip(expansion)
                label = vflip(label)
                data=torch.cat([flip_expansion, flip_img],dim=0)
            x = (data,label)
        return x
    
class SoundMultipleInstance(Dataset):
    # Single Channel Dataset 10m x 10m resolution for multiple prediction
    def __init__(self, path, index_set, multiple_factor, t_mean=None, t_std = None,transform=None):

        self.transform=transform

        self.total_img=np.load(path+'/multiple_overlap_img10_{}.npy'.format(multiple_factor))
        self.total_label=np.load(path+'/multiple_overlap_noise_{}.npy'.format(multiple_factor))
        self.total_coord=np.load(path+'/multiple_overlap_coord_{}.npy'.format(multiple_factor))   

        
        self.origin_img, self.label, self.coord = parsing_index(self.total_img, self.total_label, self.total_coord, index_set)
        self.origin_img = self.origin_img/255.0
        self.label = self.label.reshape(self.label.shape[0],multiple_factor,multiple_factor)
        
        if (t_mean is None) & (t_std is None):
            self.mean_value, self.std_value = self.get_mean_var()
        else:
            self.mean_value = t_mean
            self.std_value = t_std
        
    def get_mean_var(self):
        
        mean_value = np.mean(np.mean(self.origin_img,axis=(1,2)))
        std_value = np.mean(np.std(self.origin_img,axis=(1,2)))

        return mean_value, std_value
    
    def normalizing(self, x):
        return (x - self.mean_value)/self.std_value
            
    def __len__(self):
        return self.origin_img.shape[0]

    def __getitem__(self, index):

        img=torch.tensor(self.origin_img[index],dtype=torch.float).unsqueeze(0)
        target=torch.tensor(self.label[index],dtype=torch.float).unsqueeze(0)

        # 이 부분 다시 짜야함. 
        if self.transform is not None:
            img, target = self.transform((img,target))

        img = self.normalizing(img)
        target = target.reshape(-1)

        return img, target, index


class SoundMultipleTwoInstance(Dataset):
    # Single Channel Dataset 10m x 10m resolution for multiple prediction
    def __init__(self, path, index_set, multiple_factor, t_mean=None, t_std = None,transform=None):

        self.transform=transform

        self.total_img=np.load(path+'/multiple_overlap_img10_{}.npy'.format(multiple_factor))
        self.total_eimg=np.load(path+'/multiple_overlap_img1_{}.npy'.format(multiple_factor))
        self.total_label=np.load(path+'/multiple_overlap_noise_{}.npy'.format(multiple_factor))
        self.total_coord=np.load(path+'/multiple_overlap_coord_{}.npy'.format(multiple_factor))   

        
        self.expansion_img, self.origin_img, self.label, self.coord = parsing_index(self.total_img, self.total_label, self.total_coord, index_set, self.total_eimg)
        self.expansion_img = self.expansion_img / 255.0
        self.origin_img = self.origin_img / 255.0
        self.label = self.label.reshape(self.label.shape[0],multiple_factor,multiple_factor)
        
        if (t_mean is None) & (t_std is None):
            self.mean_value, self.std_value = self.get_mean_var()
        else:
            self.mean_value = t_mean
            self.std_value = t_std
        
    def get_mean_var(self):
        mean_value0 = np.mean(np.mean(self.expansion_img,axis=(1,2)))
        std_value0 = np.mean(np.std(self.expansion_img,axis=(1,2)))
        
        mean_value1 = np.mean(np.mean(self.origin_img,axis=(1,2)))
        std_value1 = np.mean(np.std(self.origin_img,axis=(1,2)))

        mean_vec = torch.tensor((mean_value0,mean_value1),dtype=torch.float).unsqueeze(1).unsqueeze(1)
        std_vec = torch.tensor((std_value0,std_value1),dtype=torch.float).unsqueeze(1).unsqueeze(1)

        return mean_vec, std_vec
        
    def normalizing(self, x):
        return (x - self.mean_value)/self.std_value
            
    def __len__(self):
        return self.origin_img.shape[0]

    def __getitem__(self, index):
        one_img = self.origin_img[index].reshape(1,100,100)
        expansion_img = self.expansion_img[index].reshape(1,100,100) 

        origin_img=torch.tensor(one_img,dtype=torch.float)
        expansion_img=torch.tensor(expansion_img,dtype=torch.float)
        target=torch.tensor(self.label[index],dtype=torch.float).unsqueeze(0)

        img = torch.cat([expansion_img, origin_img],dim=0)

        # 이 부분 다시 짜야함. 
        if self.transform is not None:
            img, target = self.transform((img,target))

        img = self.normalizing(img)
        target = target.reshape(-1)

        return img, target, index
    
# get 함수 짜기.
def get_overlap_multiple_sound_dataloaders(path,batch_size=128, num_workers=4, seed=0, multiple_factor=3):
    
    """
    Sound Map data efficient aug Experiment for single channel dataset
    """
    train_idx, valid_idx = choose_region(seed)
    train_transform = transforms.Compose([
        SoundMultiRandomHorizontalFlip(mtype = 'single'),
        SoundMultiRandomVerticalFlip(mtype = 'single')
        #transforms.Normalize((0.90865), (0.21802)),
    ])
    
    # test_transform = transforms.Compose([
    #     transforms.Normalize((0.90865), (0.21802)),
    # ])

    train_set = SoundMultipleInstance(path, train_idx, multiple_factor, transform = train_transform)
    n_data = len(train_set)
    
    t_mean = train_set.mean_value
    t_std = train_set.std_value

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = SoundMultipleInstance(path, valid_idx, multiple_factor, t_mean=t_mean, t_std = t_std)

    test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=num_workers)


    return train_loader, test_loader, n_data



def get_overlap_multiple_two_sound_dataloaders(path,batch_size=128, num_workers=4, seed=0, multiple_factor=3):
    
    """
    Sound Map data efficient aug Experiment for single channel dataset
    """
    train_idx, valid_idx = choose_region(seed)
    train_transform = transforms.Compose([
        SoundMultiRandomHorizontalFlip(mtype = 'two'),
        SoundMultiRandomVerticalFlip(mtype = 'two')
        #transforms.Normalize((0.90865), (0.21802)),
    ])
    
    # test_transform = transforms.Compose([
    #     transforms.Normalize((0.90865), (0.21802)),
    # ])

    train_set = SoundMultipleTwoInstance(path, train_idx, multiple_factor, transform = train_transform)
    n_data = len(train_set)
    
    t_mean = train_set.mean_value
    t_std = train_set.std_value

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = SoundMultipleTwoInstance(path, valid_idx, multiple_factor, t_mean=t_mean, t_std = t_std)

    test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=num_workers)


    return train_loader, test_loader, n_data