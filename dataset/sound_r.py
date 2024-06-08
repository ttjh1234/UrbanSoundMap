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

def parsing_index(img, label, coord, index):
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
    
    img = img[data_index]
    label = label[data_index]
    coord = coord[data_index]
    return img, label, coord

def parsing_expansion(img1, coord, index):
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

    img1 = img1[data_index]
    
    return img1


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
 


class SoundRInstance(Dataset):
    # Multi + Expansion Channel Dataset 10m x 10m, 1m x 1m resolution
    def __init__(self, path, index_set, t_mean=None, t_std = None,transform=None,target_transform=None):

        self.transform=transform
        self.target_transform=target_transform
        
        self.total_img=np.load(path+'/total_r_img.npy')
        self.train_label=np.load(path+'/train_label.npy')
        self.train_coord=np.load(path+'/train_coord.npy')    
        self.test_label=np.load(path+'/test_label.npy')
        self.test_coord=np.load(path+'/test_coord.npy')          

        self.total_label = np.concatenate([self.train_label,self.test_label],axis=0)
        self.total_coord = np.concatenate([self.train_coord,self.test_coord],axis=0)

        del self.train_label
        del self.train_coord
        del self.test_label
        del self.test_coord

        self.img, self.label, self.coord = parsing_index(self.total_img, self.total_label, self.total_coord, index_set)
        
        if (t_mean is None) & (t_std is None):
            self.mean_value, self.std_value = self.get_mean_std()
        else:
            self.mean_value = t_mean
            self.std_value = t_std
            
    def __len__(self):
        return self.img.shape[0]


    def get_mean_std(self):
        # 4 channel (N, 4, 100,100)
        
        mean_value = np.mean(np.mean(self.img,axis=(2,3)),axis=0) # 4
        std_value = np.mean(np.std(self.img,axis=(2,3)),axis=0) #4

        mean_vec = torch.tensor((mean_value),dtype=torch.float).unsqueeze(1).unsqueeze(1)
        std_vec = torch.tensor((std_value),dtype=torch.float).unsqueeze(1).unsqueeze(1)

        return mean_vec, std_vec

    
    def normalizing(self, x):
        return (x - self.mean_value)/self.std_value

    def __getitem__(self, index):
        
        img=torch.tensor(self.img[index],dtype=torch.float)
        target=torch.tensor(self.label[index],dtype=torch.float)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        img = self.normalizing(img)
        
        return img, target, index


class SoundRValidInstance(Dataset):
    # Multi + Expansion Channel Dataset 10m x 10m, 1m x 1m resolution
    def __init__(self, path, index_set,atype, istrain=False,t_mean=None, t_std = None,transform=None,target_transform=None):

        self.transform=transform
        self.target_transform=target_transform
        
        self.total_img=np.load(path+'/total_r_img.npy')
        self.train_label=np.load(path+'/train_label.npy')
        self.train_coord=np.load(path+'/train_coord.npy')    
        self.test_label=np.load(path+'/test_label.npy')
        self.test_coord=np.load(path+'/test_coord.npy')          

        self.total_label = np.concatenate([self.train_label,self.test_label],axis=0)
        self.total_coord = np.concatenate([self.train_coord,self.test_coord],axis=0)

        del self.train_label
        del self.train_coord
        del self.test_label
        del self.test_coord
        
        self.img, self.label, self.coord = parsing_index(self.total_img, self.total_label, self.total_coord, index_set)
        
        if istrain==False:
            self.train_exp = np.load(path+'/train_img1.npy')
            self.test_exp = np.load(path+'/test_img1.npy')
            
            self.expansion_img = np.concatenate([self.train_exp, self.test_exp],axis=0)
            del self.train_exp
            del self.test_exp
        
            self.eimg = parsing_expansion(self.expansion_img, self.total_coord, index_set)
        
        if (t_mean is None) & (t_std is None):
            self.mean_value, self.std_value = self.get_mean_std()
        else:
            self.mean_value = t_mean
            self.std_value = t_std
        
        if istrain==False:
            self.img, self.label= self.dataset_split(self.eimg, self.img, self.label, atype=atype)
            
    def __len__(self):
        return self.img.shape[0]

    def dataset_split(self,eimg, oimg, label, atype):
        if atype =='total':
            return oimg, label
        
        elif atype =='center':
            build = np.where(eimg<=90/255.0, eimg, 1.0)
            road = np.where(eimg>=90/255.0, eimg, 1.0)
            road_flag = np.where(np.sum(1-road,axis=(1,2))!=0, 1, 0)
            build_flag = np.where(np.sum(1-build,axis=(1,2))!=0, 1, 0)

            mask = (road_flag==1) & (build_flag==1)
            bothin = np.where(mask)[0] # Center idx

            # center
            oimg = oimg[bothin]
            label = label[bothin]
            return oimg, label
            
        elif atype == 'noncenter':
            build = np.where(eimg<=90/255.0, eimg, 1.0)
            road = np.where(eimg>=90/255.0, eimg, 1.0)
            road_flag = np.where(np.sum(1-road,axis=(1,2))!=0, 1, 0)
            build_flag = np.where(np.sum(1-build,axis=(1,2))!=0, 1, 0)

            mask = (road_flag==0) | (build_flag==0)
            bothnotin = np.where(mask)[0] # Center idx

            # center
            oimg = oimg[bothnotin]
            label = label[bothnotin]
            return oimg, label
        else:
            raise NotImplementedError()


    def get_mean_std(self):
        # 4 channel (N, 4, 100,100)
        
        mean_value = np.mean(np.mean(self.img,axis=(2,3)),axis=0) # 4
        std_value = np.mean(np.std(self.img,axis=(2,3)),axis=0) #4

        mean_vec = torch.tensor((mean_value),dtype=torch.float).unsqueeze(1).unsqueeze(1)
        std_vec = torch.tensor((std_value),dtype=torch.float).unsqueeze(1).unsqueeze(1)

        return mean_vec, std_vec

    
    def normalizing(self, x):
        return (x - self.mean_value)/self.std_value

    def __getitem__(self, index):
        
        img=torch.tensor(self.img[index],dtype=torch.float)
        target=torch.tensor(self.label[index],dtype=torch.float)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        img = self.normalizing(img)
        
        return img, target, index
    

def get_r_sound_dataloaders(path,batch_size=128, num_workers=4, seed=0):
    
    """
    Sound Map multi + expansion data
    """
    train_idx, valid_idx = choose_region(seed)
    train_transform = transforms.Compose([
        SoundRandomHorizontalFlip(),
        SoundRandomVerticalFlip(),
    ])


    train_set = SoundRInstance(path,train_idx, transform = train_transform)
    n_data = len(train_set)
    
    t_mean = train_set.mean_value
    t_std = train_set.std_value

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = SoundRInstance(path,valid_idx, t_mean=t_mean, t_std = t_std)

    test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=num_workers)

    return train_loader, test_loader, n_data

def get_r_valid_sound_dataloaders(path, atype='total', batch_size=128, num_workers=4, seed=0):
    
    """
    Sound Map multi + expansion data
    """
    train_idx, valid_idx = choose_region(seed)

    train_set = SoundRValidInstance(path,train_idx, atype=atype, istrain=True)

    t_mean = train_set.mean_value
    t_std = train_set.std_value

    test_set = SoundRValidInstance(path,valid_idx, atype=atype, t_mean=t_mean, t_std = t_std)

    test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=num_workers)

    return test_loader

