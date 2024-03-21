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
 
class SoundInstance(Dataset):
    # Single Channel Dataset 10m x 10m resolution
    def __init__(self, path, index_set, train=True, transform=None,target_transform=None):

        self.transform=transform
        self.target_transform=target_transform
        self.train=train
        
        if self.train:
            self.origin_img=np.load(path+'/train_img10.npy')
            self.label=np.load(path+'/train_label.npy')
        
        else:
            self.origin_img=np.load(path+'/test_img10.npy')
            self.label=np.load(path+'/test_label.npy')    
            
    def __len__(self):
        return self.origin_img.shape[0]

    def __getitem__(self, index):

        img=torch.tensor(self.origin_img[index],dtype=torch.float).unsqueeze(0)
        target=torch.tensor(self.label[index],dtype=torch.float)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class SoundExpansionInstance(Dataset):
    # Single Channel Dataset 1m x 1m resolution
    def __init__(self, path, train=True, transform=None,target_transform=None):

        self.transform=transform
        self.target_transform=target_transform
        self.train=train
               
        if self.train:
            self.img=np.load(path+'/train_img1.npy')           
            self.label=np.load(path+'/train_label.npy')
        
        else:
            self.img=np.load(path+'/test_img1.npy')
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


class SoundDetachedOriginInstance(Dataset):
    # Multi Channel Dataset 10m x 10m resolution
    def __init__(self, path, train=True, transform=None,target_transform=None):

        self.transform=transform
        self.target_transform=target_transform
        self.train=train
               
        if self.train:
            self.origin_img=np.load(path+'/train_img10.npy')
            self.label=np.load(path+'/train_label.npy')
        
        else:
            self.origin_img=np.load(path+'/test_img10.npy')
            self.label=np.load(path+'/test_label.npy')

            
    def __len__(self):
        return self.origin_img.shape[0]

    def __getitem__(self, index):
        one_img = self.origin_img[index].reshape(1,100,100)
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
    def __init__(self, path, train=True, transform=None,target_transform=None):

        self.transform=transform
        self.target_transform=target_transform
        self.train=train
               
        if self.train:
            self.origin_img=np.load(path+'/train_img10.npy')
            self.expansion_img=np.load(path+'/train_img1.npy')           
            self.label=np.load(path+'/train_label.npy')
        
        else:
            self.origin_img=np.load(path+'/test_img10.npy')
            self.expansion_img=np.load(path+'/test_img1.npy')           
            self.label=np.load(path+'/test_label.npy')

            
    def __len__(self):
        return self.origin_img.shape[0]

    def __getitem__(self, index):
        one_img = self.origin_img[index].reshape(1,100,100)
        expansion_img = self.expansion_img[index].reshape(1,100,100) 
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


class SoundDetachedExpressInstance(Dataset):
    # Multi + Expansion Channel Dataset 10m x 10m, 1m x 1m resolution, delete non-object image
    def __init__(self, path, train=True, transform=None,target_transform=None):

        self.transform=transform
        self.target_transform=target_transform
        self.train=train
               
        if self.train:
            self.origin_img=np.load(path+'/train_img10.npy')
            self.expansion_img=np.load(path+'/train_img1.npy')           
            self.label=np.load(path+'/train_label.npy')
            self.origin_img = np.delete(self.origin_img,[89300,89313,149813],axis=0)
            self.expansion_img = np.delete(self.expansion_img,[89300,89313,149813],axis=0)
            self.label = np.delete(self.label,[89300,89313,149813],axis=0)
        
        else:
            self.origin_img=np.load(path+'/test_img10.npy')
            self.expansion_img=np.load(path+'/test_img1.npy')           
            self.label=np.load(path+'/test_label.npy')

            
    def __len__(self):
        return self.origin_img.shape[0]

    def __getitem__(self, index):
        one_img = self.origin_img[index].reshape(1,100,100)
        expansion_img = self.expansion_img[index].reshape(1,100,100) 
        if np.isclose(np.sum(1-expansion_img,axis=(1,2)),0.0, atol=1e-5):
            expansion_img = one_img
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


class SoundDetachedMaskInstance(Dataset):
    # Multi Channel Dataset 10m x 10m + Mask generate
    def __init__(self, path, train=True, transform=None,target_transform=None):

        self.transform=transform
        self.target_transform=target_transform
        self.train=train
               
        if self.train:
            self.origin_img=np.load(path+'/train_img10.npy')
            #self.expansion_img=np.load(path+'/train_img1.npy')           
            self.label=np.load(path+'/train_label.npy')
        
        else:
            self.origin_img=np.load(path+'/test_img10.npy')
            #self.expansion_img=np.load(path+'/test_img1.npy')           
            self.label=np.load(path+'/test_label.npy')

            
    def __len__(self):
        return self.origin_img.shape[0]

    def __getitem__(self, index):
        one_img = self.origin_img[index].reshape(1,100,100)
        #expansion_img = self.expansion_img[index].reshape(1,100,100) 
        multi_channel_img = np.concatenate([np.ones((2,100,100)),one_img],axis=0)

        multi_channel_img[0,:,:] = np.where(multi_channel_img[2]<=90/255.0, multi_channel_img[2],1.0)
        multi_channel_img[1,:,:] = np.where(multi_channel_img[2]>90/255.0, multi_channel_img[2],1.0)
        
        mask = np.where(multi_channel_img[2]==1.0, 0.0, 1.0).reshape(1,100,100) # 1,100,100
        
        #origin_img=torch.tensor(multi_channel_img,dtype=torch.float)
        #expansion_img=torch.tensor(expansion_img,dtype=torch.float)
        img=torch.tensor(multi_channel_img,dtype=torch.float)
        target=torch.tensor(self.label[index],dtype=torch.float)

        #img = torch.cat([expansion_img, origin_img],dim=0)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, mask, target, index

def get_no_aug_sound_dataloaders(path,batch_size=128, num_workers=4):
    
    """
    Sound Map data No aug Experiment for single channel dataset
    """
    
    train_transform = transforms.Compose([
        transforms.Normalize((0.91374), (0.20895)),
    ])
    
    test_transform = transforms.Compose([
        transforms.Normalize((0.91374), (0.20895)),
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

def get_aug_crop_sound_dataloaders(path,batch_size=128, num_workers=4):
    
    """
    Sound Map data not efficient aug Experiment for single channel dataset
    """
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(100,4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize((0.91374), (0.20895)),
    ])
    
    test_transform = transforms.Compose([
        transforms.Normalize((0.91374), (0.20895)),
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

def get_sound_dataloaders(path,batch_size=128, num_workers=4):
    
    """
    Sound Map data efficient aug Experiment for single channel dataset
    """
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize((0.91374), (0.20895)),
    ])
    
    test_transform = transforms.Compose([
        transforms.Normalize((0.91374), (0.20895)),
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



def get_expansion_sound_dataloaders(path,batch_size=128, num_workers=4):
    
    """
    Sound Map Expansion data (1m x 1m ) Only, single channel
    """
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize((0.93444), (0.13224)),
    ])
    
    test_transform = transforms.Compose([
        transforms.Normalize((0.93444), (0.13224)),
    ])


    train_set = SoundExpansionInstance(path,train = True, transform = train_transform)
    n_data = len(train_set)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = SoundExpansionInstance(path, train=False, transform = test_transform)

    test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=num_workers)


    return train_loader, test_loader, n_data

def get_detached_origin_sound_dataloaders(path,batch_size=128, num_workers=4):
    
    """
    Sound Map multi channel data
    """
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize((0.92018, 0.99356 ,0.91374), (0.20518, 0.04393 ,0.20895)),
    ])
    
    test_transform = transforms.Compose([
        transforms.Normalize((0.92018, 0.99356 ,0.91374), (0.20518, 0.04393 ,0.20895)),
    ])


    train_set = SoundDetachedOriginInstance(path,train = True, transform = train_transform)
    n_data = len(train_set)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = SoundDetachedOriginInstance(path, train=False, transform = test_transform)

    test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=num_workers)


    return train_loader, test_loader, n_data

def get_detached_sound_dataloaders(path,batch_size=128, num_workers=4):
    
    """
    Sound Map multi + expansion data
    """
    
    train_transform = transforms.Compose([
        SoundRandomHorizontalFlip(),
        SoundRandomVerticalFlip(),
        transforms.Normalize((0.93444, 0.92018, 0.99356 ,0.91374), (0.13224, 0.20518, 0.04393 ,0.20895)),
    ])
    
    test_transform = transforms.Compose([
        transforms.Normalize((0.93444, 0.92018, 0.99356 ,0.91374), (0.13224, 0.20518, 0.04393 ,0.20895)),
    ])


    train_set = SoundDetachedInstance(path,train = True, transform = train_transform)
    n_data = len(train_set)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = SoundDetachedInstance(path, train=False, transform = test_transform)

    test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=num_workers)


    return train_loader, test_loader, n_data


def get_detached_express_sound_dataloaders(path,batch_size=128, num_workers=4):
    
    """
    Sound Map data
    """
    
    train_transform = transforms.Compose([
        SoundRandomHorizontalFlip(),
        SoundRandomVerticalFlip(),
        transforms.Normalize((0.92119, 0.92018, 0.99356 ,0.91374), (0.18750, 0.20518, 0.04393 ,0.20895)),
    ])
    
    test_transform = transforms.Compose([
        transforms.Normalize((0.92119, 0.92018, 0.99356 ,0.91374), (0.18750, 0.20518, 0.04393 ,0.20895)),
    ])


    train_set = SoundDetachedExpressInstance(path,train = True, transform = train_transform)
    n_data = len(train_set)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = SoundDetachedExpressInstance(path, train=False, transform = test_transform)

    test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=num_workers)


    return train_loader, test_loader, n_data

def get_detached_mask_sound_dataloaders(path,batch_size=128, num_workers=4):
    
    """
    Sound Map multi channel data with mask
    """
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize((0.92018, 0.99356 ,0.91374), (0.20518, 0.04393 ,0.20895)),
    ])
    
    test_transform = transforms.Compose([
        transforms.Normalize((0.92018, 0.99356 ,0.91374), (0.20518, 0.04393 ,0.20895)),
    ])


    train_set = SoundDetachedMaskInstance(path,train = True, transform = train_transform)
    n_data = len(train_set)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = SoundDetachedMaskInstance(path, train=False, transform = test_transform)

    test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=num_workers)


    return train_loader, test_loader, n_data


def get_detached_sound_noaug_dataloaders(path,batch_size=128, num_workers=4):
    
    """
    Sound Map multi + expansion data, no aug version
    """
    
    train_transform = transforms.Compose([
        transforms.Normalize((0.93444, 0.92018, 0.99356 ,0.91374), (0.13224, 0.20518, 0.04393 ,0.20895)),
    ])
    
    test_transform = transforms.Compose([
        transforms.Normalize((0.93444, 0.92018, 0.99356 ,0.91374), (0.13224, 0.20518, 0.04393 ,0.20895)),
    ])


    train_set = SoundDetachedInstance(path,train = True, transform = train_transform)
    n_data = len(train_set)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = SoundDetachedInstance(path, train=False, transform = test_transform)

    test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=num_workers)


    return train_loader, test_loader, n_data


