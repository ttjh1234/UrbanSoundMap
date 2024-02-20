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
from torchvision.transforms.functional import hflip, rotate
from PIL import Image
from scipy import interpolate
import random

class SoundRandomRotate(nn.Module):
    def __init__(self, candidate_list):
        super().__init__()
        self.candidate = candidate_list
    
    def forward(self, x):
        angle = random.choice(self.candidate)
        x = rotate(x, angle)

        return x


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


class SoundDetachedInstance(Dataset):
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
        
        self.prompt = np.zeros((100,100))
        self.prompt[49:51,49:51] = 0.0
        self.prompt = (self.prompt - 0.9996)/0.02
        
    def __len__(self):
        return self.img.shape[0]

    def __getitem__(self, index):
        one_img = self.img[index].reshape(1,100,100)
        multi_channel_img = np.concatenate([np.ones((2,100,100)),one_img],axis=0)

        multi_channel_img[0,:,:] = np.where(multi_channel_img[2]<=90/255.0, multi_channel_img[2],1.0)
        multi_channel_img[1,:,:] = np.where(multi_channel_img[2]>90/255.0, multi_channel_img[2],1.0)
        
        img = torch.tensor(multi_channel_img,dtype=torch.float)
        target = torch.tensor(self.label[index],dtype=torch.float)
        prompt = torch.tensor(self.prompt, dtype= torch.float).unsqueeze(0)
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        img = torch.cat([img, prompt], dim=0)
        
        return img, target, index

class SoundDetachedMaskInstance(Dataset):
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
        one_img = self.img[index].reshape(1,100,100)
        multi_channel_img = np.concatenate([np.ones((2,100,100)),one_img],axis=0)

        multi_channel_img[0,:,:] = np.where(multi_channel_img[2]<=90/255.0, multi_channel_img[2],1.0)
        multi_channel_img[1,:,:] = np.where(multi_channel_img[2]>90/255.0, multi_channel_img[2],1.0)
        mask = np.where(np.isclose(one_img,1.0,atol=1e-5),1,0)
        
        img=torch.tensor(multi_channel_img,dtype=torch.float)
        target=torch.tensor(self.label[index],dtype=torch.float)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, mask, index


class SoundDetachedInstance2(Dataset):
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

    def road_minmax(self,x):
        x = np.clip((x - 90/255) / ((255 - 90)/255),0.0,1.0)
        return x

    def build_minmax(self,x):
        x = x/(90/255.0)
        return x

    def __getitem__(self, index):
        # one_img = self.img[index].reshape(1,100,100)
        # multi_channel_img = np.concatenate([np.ones((2,100,100)),one_img],axis=0)

        # multi_channel_img[0,:,:] = np.where(multi_channel_img[2]<=90/255.0, multi_channel_img[2],0.0)
        # multi_channel_img[1,:,:] = np.where(multi_channel_img[2]>90/255.0, multi_channel_img[2],0.0)
        # multi_channel_img[1,:,:] = np.where(multi_channel_img[1,:,:]>=1.0, 0.0, multi_channel_img[1,:,:])
        # multi_channel_img[0,:,:] = self.build_minmax(multi_channel_img[0,:,:])
        # multi_channel_img[1,:,:] = self.road_minmax(multi_channel_img[1,:,:])
        
        # img=torch.tensor(multi_channel_img,dtype=torch.float)
        # target=torch.tensor(self.label[index],dtype=torch.float)

        # if self.transform is not None:
        #     img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        # return img, target, index
        one_img = self.img[index].reshape(1,100,100)
        multi_channel_img = np.concatenate([np.ones((2,100,100)),one_img],axis=0)

        multi_channel_img[0,:,:] = np.where(multi_channel_img[2]<=90/255.0, multi_channel_img[2],0.0)
        multi_channel_img[1,:,:] = np.where(multi_channel_img[2]>90/255.0, multi_channel_img[2],0.0)
        multi_channel_img[1,:,:] = np.where(multi_channel_img[1,:,:]>=1.0, 0.0, multi_channel_img[1,:,:])
        multi_channel_img[0,:,:] = self.build_minmax(multi_channel_img[0,:,:])
        multi_channel_img[1,:,:] = self.road_minmax(multi_channel_img[1,:,:])
        
        img=torch.tensor(multi_channel_img[:2,:,:],dtype=torch.float)
        target=torch.tensor(self.label[index],dtype=torch.float)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class SoundDetachedInstance3(Dataset):
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

    def road_minmax(self,x):
        x = np.clip((x - 90/255) / ((255 - 90)/255),0.0,1.0)
        return x

    def build_minmax(self,x):
        x = x/(90/255.0)
        return x

    def __getitem__(self, index):
        # one_img = self.img[index].reshape(1,100,100)
        # multi_channel_img = np.concatenate([np.ones((2,100,100)),one_img],axis=0)

        # multi_channel_img[0,:,:] = np.where(multi_channel_img[2]<=90/255.0, multi_channel_img[2],0.0)
        # multi_channel_img[1,:,:] = np.where(multi_channel_img[2]>90/255.0, multi_channel_img[2],0.0)
        # multi_channel_img[1,:,:] = np.where(multi_channel_img[1,:,:]>=1.0, 0.0, multi_channel_img[1,:,:])
        # multi_channel_img[0,:,:] = self.build_minmax(multi_channel_img[0,:,:])
        # multi_channel_img[1,:,:] = self.road_minmax(multi_channel_img[1,:,:])
        
        # img=torch.tensor(multi_channel_img,dtype=torch.float)
        # target=torch.tensor(self.label[index],dtype=torch.float)

        # if self.transform is not None:
        #     img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        # return img, target, index
        one_img = self.img[index].reshape(1,100,100)
        multi_channel_img = np.concatenate([np.ones((2,100,100)),one_img],axis=0)

        multi_channel_img[0,:,:] = np.where(multi_channel_img[2]<=90/255.0, multi_channel_img[2],0.0)
        multi_channel_img[1,:,:] = np.where(multi_channel_img[2]>90/255.0, multi_channel_img[2],0.0)
        multi_channel_img[1,:,:] = np.where(multi_channel_img[1,:,:]>=1.0, 0.0, multi_channel_img[1,:,:])
        multi_channel_img[0,:,:] = self.build_minmax(multi_channel_img[0,:,:])
        multi_channel_img[1,:,:] = self.road_minmax(multi_channel_img[1,:,:])
        
        img=torch.tensor(multi_channel_img,dtype=torch.float)
        target=torch.tensor(self.label[index],dtype=torch.float)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class SoundDetachedTwoInstance(Dataset):
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
        one_img = self.img[index].reshape(1,100,100)
        multi_channel_img = np.concatenate([np.ones((2,100,100)),one_img],axis=0)

        multi_channel_img[0,:,:] = np.where(multi_channel_img[2]<=90/255.0, multi_channel_img[2],1.0)
        multi_channel_img[1,:,:] = np.where(multi_channel_img[2]>90/255.0, multi_channel_img[2],1.0)
        
        img=torch.tensor(multi_channel_img[:2,:,:],dtype=torch.float)
        target=torch.tensor(self.label[index],dtype=torch.float)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

class SoundAnalysis(Dataset):
    def __init__(self, path, train=True, transform=None,target_transform=None):

        self.transform=transform
        self.target_transform=target_transform
        self.train=train
               
        if self.train:
            self.img=np.load(path+'/train_img.npy')
            self.label=np.load(path+'/train_label.npy')
            self.coord=np.load(path+'/train_coord.npy')
        
        else:
            self.img=np.load(path+'/test_img.npy')
            self.label=np.load(path+'/test_label.npy')
            self.coord=np.load(path+'/test_coord.npy')
            
    def __len__(self):
        return self.img.shape[0]

    def __getitem__(self, index):

        img=torch.tensor(self.img[index],dtype=torch.float).unsqueeze(0)
        target=torch.tensor(self.label[index],dtype=torch.float)
        coord=torch.tensor(self.coord[index],dtype=torch.float)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, coord, index

    

def get_sound_dataloaders(path,batch_size=128, num_workers=4):
    
    """
    Sound Map data
    """
    
    # train_transform = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomVerticalFlip(),
    #     SoundRandomRotate([0,90,180,270]),
    #     transforms.Normalize((0.91374), (0.20895)),
    # ])
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

def get_sound_dataloaders_no_aug(path,batch_size=128, num_workers=4):
    
    """
    Sound Map data
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


def get_sound_analysis_dataloaders(path,batch_size=128, num_workers=4):
    """
    Sound Map data
    """
    
    train_transform = transforms.Compose([
        transforms.Normalize((0.91374), (0.20895)),
    ])
    
    test_transform = transforms.Compose([
        transforms.Normalize((0.91374), (0.20895)),
    ])


    train_set = SoundAnalysis(path,train = True, transform = train_transform)
    n_data = len(train_set)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers)

    test_set = SoundAnalysis(path, train=False, transform = test_transform)

    test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=num_workers)


    return train_loader, test_loader, n_data


def get_detached_sound_dataloaders(path,batch_size=128, num_workers=4):
    
    """
    Sound Map data
    """
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize(( 0.92018, 0.99356 ,0.91374), (0.20518, 0.04393 ,0.20895)),
    ])
    
    test_transform = transforms.Compose([
        transforms.Normalize(( 0.92018, 0.99356 ,0.91374), (0.20518, 0.04393 ,0.20895)),
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

def get_detached_sound_noaug_dataloaders(path,batch_size=128, num_workers=4):
    
    """
    Sound Map data
    """
    
    train_transform = transforms.Compose([
        transforms.Normalize(( 0.92018, 0.99356 ,0.91374), (0.20518, 0.04393 ,0.20895)),
    ])
    
    test_transform = transforms.Compose([
        transforms.Normalize(( 0.92018, 0.99356 ,0.91374), (0.20518, 0.04393 ,0.20895)),
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


def get_detached_mask_sound_dataloaders(path,batch_size=128, num_workers=4):
    
    """
    Sound Map data
    """
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize(( 0.92018, 0.99356 ,0.91374), (0.20518, 0.04393 ,0.20895)),
    ])
    
    test_transform = transforms.Compose([
        transforms.Normalize(( 0.92018, 0.99356 ,0.91374), (0.20518, 0.04393 ,0.20895)),
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


def get_detached_sound_dataloaders2(path,batch_size=128, num_workers=4):
    
    """
    Sound Map data
    """
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize((0.03063, 0.00889), (0.08858, 0.06047)),
    ])
    
    test_transform = transforms.Compose([
        transforms.Normalize((0.03063, 0.00889), (0.08858, 0.06047)),
    ])


    train_set = SoundDetachedInstance2(path,train = True, transform = train_transform)
    n_data = len(train_set)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = SoundDetachedInstance2(path, train=False, transform = test_transform)

    test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=num_workers)


    return train_loader, test_loader, n_data


def get_detached_sound_dataloaders3(path,batch_size=128, num_workers=4):
    
    """
    Sound Map data
    """
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize((0.03063, 0.00889,0.91374), (0.08858, 0.06047,0.20895)),
    ])
    
    test_transform = transforms.Compose([
        transforms.Normalize((0.03063, 0.00889,0.91374), (0.08858, 0.06047,0.20895)),
    ])


    train_set = SoundDetachedInstance3(path,train = True, transform = train_transform)
    n_data = len(train_set)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = SoundDetachedInstance3(path, train=False, transform = test_transform)

    test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=num_workers)


    return train_loader, test_loader, n_data


def get_detached_two_sound_dataloaders(path,batch_size=128, num_workers=4):
    
    """
    Sound Map data
    """
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize(( 0.92018, 0.99356), (0.20518, 0.04393)),
    ])
    
    test_transform = transforms.Compose([
        transforms.Normalize(( 0.92018, 0.99356), (0.20518, 0.04393)),
    ])


    train_set = SoundDetachedTwoInstance(path,train = True, transform = train_transform)
    n_data = len(train_set)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = SoundDetachedTwoInstance(path, train=False, transform = test_transform)

    test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=num_workers)


    return train_loader, test_loader, n_data