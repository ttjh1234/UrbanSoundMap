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


def parsing_index_multi(eximg1, eximg2, img10, label, coord, index):
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

    img10 = img10[data_index]
    eximg1 = eximg1[data_index]
    eximg2 = eximg2[data_index]
    label = label[data_index]
    coord = coord[data_index]
    return eximg1,eximg2, img10, label, coord

def parsing_index_four(eximg1, eximg2, eximg3, img10, label, coord, index):
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

    img10 = img10[data_index]
    eximg1 = eximg1[data_index]
    eximg2 = eximg2[data_index]
    eximg3 = eximg3[data_index]
    label = label[data_index]
    coord = coord[data_index]
    return eximg1, eximg2, eximg3, img10, label, coord


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
    def __init__(self, path, index_set, t_mean=None, t_std = None,transform=None,target_transform=None):

        self.transform=transform
        self.target_transform=target_transform

        self.train_img=np.load(path+'/train_img10.npy')
        self.train_label=np.load(path+'/train_label.npy')
        self.train_coord=np.load(path+'/train_coord.npy')    
        self.test_img=np.load(path+'/test_img10.npy')
        self.test_label=np.load(path+'/test_label.npy')
        self.test_coord=np.load(path+'/test_coord.npy')    
        
        self.total_img = np.concatenate([self.train_img,self.test_img],axis=0)
        self.total_label = np.concatenate([self.train_label,self.test_label],axis=0)
        self.total_coord = np.concatenate([self.train_coord,self.test_coord],axis=0)
        
        self.origin_img, self.label, self.coord = parsing_index(self.total_img, self.total_label, self.total_coord, index_set)
        
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
        target=torch.tensor(self.label[index],dtype=torch.float)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        img = self.normalizing(img)

        return img, target, index


class SoundTwoChannelInstance(Dataset):
    # Multi + Expansion Channel Dataset 10m x 10m, 1m x 1m resolution
    def __init__(self, path, index_set, expansion, t_mean=None, t_std = None,transform=None,target_transform=None):

        self.transform=transform
        self.target_transform=target_transform
        
        self.train_img10=np.load(path+'/train_img10.npy')
        self.train_label=np.load(path+'/train_label.npy')
        self.train_coord=np.load(path+'/train_coord.npy')    
        self.test_img10=np.load(path+'/test_img10.npy')
        self.test_label=np.load(path+'/test_label.npy')
        self.test_coord=np.load(path+'/test_coord.npy')    
        self.total_expand = np.load(path+'./total_img{}.npy'.format(expansion))

        self.total_img10 = np.concatenate([self.train_img10,self.test_img10],axis=0)

        del self.train_img10
        del self.test_img10

        self.total_label = np.concatenate([self.train_label,self.test_label],axis=0)
        self.total_coord = np.concatenate([self.train_coord,self.test_coord],axis=0)

        del self.train_label
        del self.train_coord
        del self.test_label
        del self.test_coord

        self.expansion_img, self.origin_img, self.label, self.coord = parsing_index(self.total_img10, 
                                                                                    self.total_label, 
                                                                                    self.total_coord, 
                                                                                    index_set, 
                                                                                    self.total_expand)
        self.expansion_img = self.expansion_img/255.0
        
        if (t_mean is None) & (t_std is None):
            self.mean_value, self.std_value = self.get_mean_var()
        else:
            self.mean_value = t_mean
            self.std_value = t_std
            
    def __len__(self):
        return self.origin_img.shape[0]


    def get_mean_var(self):
        # 3 channel         
        mean_value0 = np.mean(np.mean(self.expansion_img,axis=(1,2)))
        std_value0 = np.mean(np.std(self.expansion_img,axis=(1,2)))
        
        mean_value1 = np.mean(np.mean(self.origin_img,axis=(1,2)))
        std_value1 = np.mean(np.std(self.origin_img,axis=(1,2)))

        mean_vec = torch.tensor((mean_value0,mean_value1),dtype=torch.float).unsqueeze(1).unsqueeze(1)
        std_vec = torch.tensor((std_value0,std_value1),dtype=torch.float).unsqueeze(1).unsqueeze(1)

        return mean_vec, std_vec

    def normalizing(self, x):
        return (x - self.mean_value)/self.std_value


    def __getitem__(self, index):
        one_img = self.origin_img[index].reshape(1,100,100)
        expansion_img = self.expansion_img[index].reshape(1,100,100) 

        origin_img=torch.tensor(one_img,dtype=torch.float)
        expansion_img=torch.tensor(expansion_img,dtype=torch.float)
        target=torch.tensor(self.label[index],dtype=torch.float)

        img = torch.cat([expansion_img, origin_img],dim=0)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        img = self.normalizing(img)
        
        return img, target, index



class SoundThreeChannelInstance(Dataset):
    # Multi Channel Dataset 10m x 10m resolution
    def __init__(self, path, index_set, expansion_list, t_mean=None, t_std = None, transform=None,target_transform=None):
            
        self.transform=transform
        self.target_transform=target_transform
                
        self.train_img=np.load(path+'/train_img10.npy')
        self.train_label=np.load(path+'/train_label.npy')
        self.train_coord=np.load(path+'/train_coord.npy')    
        self.test_img=np.load(path+'/test_img10.npy')
        self.test_label=np.load(path+'/test_label.npy')
        self.test_coord=np.load(path+'/test_coord.npy')    
        
        self.total_img = np.concatenate([self.train_img,self.test_img],axis=0)
        self.total_label = np.concatenate([self.train_label,self.test_label],axis=0)
        self.total_coord = np.concatenate([self.train_coord,self.test_coord],axis=0)
        self.total_expand1 = np.load(path+'/total_img{}.npy'.format(expansion_list[0]))
        self.total_expand2 = np.load(path+'/total_img{}.npy'.format(expansion_list[1]))

        self.expand_img1, self.expand_img2, self.origin_img, self.label, self.coord = parsing_index_multi(self.total_expand1, self.total_expand2, self.total_img, 
                                                                      self.total_label, self.total_coord, index_set)
        
        self.expand_img1 = self.expand_img1/255.0
        self.expand_img2 = self.expand_img2/255.0
        
        if (t_mean is None) & (t_std is None):
            self.mean_value, self.std_value = self.get_mean_var()
        else:
            self.mean_value = t_mean
            self.std_value = t_std
        
    def __len__(self):
        return self.origin_img.shape[0]

    def get_mean_var(self):       
        mean_value1 = np.mean(np.mean(self.expand_img1,axis=(1,2)))
        std_value1 = np.mean(np.std(self.expand_img1,axis=(1,2)))
        
        mean_value2 = np.mean(np.mean(self.expand_img2,axis=(1,2)))
        std_value2 = np.mean(np.std(self.expand_img2,axis=(1,2)))
        
        mean_value3 = np.mean(np.mean(self.origin_img,axis=(1,2)))
        std_value3 = np.mean(np.std(self.origin_img,axis=(1,2)))

        mean_vec = torch.tensor((mean_value1,mean_value2, mean_value3),dtype=torch.float).unsqueeze(1).unsqueeze(1)
        std_vec = torch.tensor((std_value1,std_value2, std_value3),dtype=torch.float).unsqueeze(1).unsqueeze(1)
        return mean_vec, std_vec

    def normalizing(self, x):
        return (x - self.mean_value)/self.std_value


    def __getitem__(self, index):
        one_img = self.origin_img[index].reshape(1,100,100)
        expansion_img1 = self.expand_img1[index].reshape(1,100,100) 
        expansion_img2 = self.expand_img2[index].reshape(1,100,100) 

        origin_img=torch.tensor(one_img,dtype=torch.float)
        expansion_img1=torch.tensor(expansion_img1,dtype=torch.float)
        expansion_img2=torch.tensor(expansion_img2,dtype=torch.float)
        
        img=torch.cat([expansion_img1, expansion_img2, origin_img],dim=0)
        target=torch.tensor(self.label[index],dtype=torch.float)


        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        img = self.normalizing(img)
        
        return img, target, index


class SoundFourChannelInstance(Dataset):
    # Multi + Expansion Channel Dataset 10m x 10m, 1m x 1m resolution
    def __init__(self, path, index_set,expansion_list, t_mean=None, t_std = None,transform=None,target_transform=None):

        self.transform=transform
        self.target_transform=target_transform
        
        self.train_img10=np.load(path+'/train_img10.npy')
        self.train_label=np.load(path+'/train_label.npy')
        self.train_coord=np.load(path+'/train_coord.npy')    
        self.test_img10=np.load(path+'/test_img10.npy')
        self.test_label=np.load(path+'/test_label.npy')
        self.test_coord=np.load(path+'/test_coord.npy')    
        
        
        self.total_img = np.concatenate([self.train_img10,self.test_img10],axis=0)
        del self.train_img10
        del self.test_img10

        self.total_label = np.concatenate([self.train_label,self.test_label],axis=0)
        self.total_coord = np.concatenate([self.train_coord,self.test_coord],axis=0)

        del self.train_label
        del self.train_coord
        del self.test_label
        del self.test_coord

        self.total_expand1 = np.load(path+'/total_img{}.npy'.format(expansion_list[0]))
        self.total_expand2 = np.load(path+'/total_img{}.npy'.format(expansion_list[1]))
        self.total_expand3 = np.load(path+'/total_img{}.npy'.format(expansion_list[2]))

        self.expand_img1, self.expand_img2,self.expand_img3, self.origin_img, self.label, self.coord = parsing_index_four(self.total_expand1, self.total_expand2, 
                                                                                                                          self.total_expand3,self.total_img, 
                                                                                                                          self.total_label, self.total_coord, index_set)
        
        self.expand_img1 = self.expand_img1/255.0
        self.expand_img2 = self.expand_img2/255.0
        self.expand_img3 = self.expand_img3/255.0
        
        if (t_mean is None) & (t_std is None):
            self.mean_value, self.std_value = self.get_mean_var()
        else:
            self.mean_value = t_mean
            self.std_value = t_std
            
    def __len__(self):
        return self.origin_img.shape[0]


    def get_mean_var(self):
        # 3 channel 
        
        mean_value0 = np.mean(np.mean(self.expand_img1,axis=(1,2)))
        std_value0 = np.mean(np.std(self.expand_img1,axis=(1,2)))
        
        mean_value1 = np.mean(np.mean(self.expand_img2,axis=(1,2)))
        std_value1 = np.mean(np.std(self.expand_img2,axis=(1,2)))
        
        mean_value2 = np.mean(np.mean(self.expand_img3,axis=(1,2)))
        std_value2 = np.mean(np.std(self.expand_img3,axis=(1,2)))
        
        mean_value3 = np.mean(np.mean(self.origin_img,axis=(1,2)))
        std_value3 = np.mean(np.std(self.origin_img,axis=(1,2)))

        mean_vec = torch.tensor((mean_value0,mean_value1,mean_value2, mean_value3),dtype=torch.float).unsqueeze(1).unsqueeze(1)
        std_vec = torch.tensor((std_value0,std_value1,std_value2, std_value3),dtype=torch.float).unsqueeze(1).unsqueeze(1)

        return mean_vec, std_vec

    def normalizing(self, x):
        return (x - self.mean_value)/self.std_value


    def __getitem__(self, index):
        one_img = self.origin_img[index].reshape(1,100,100)
        expansion_img1 = self.expand_img1[index].reshape(1,100,100) 
        expansion_img2 = self.expand_img2[index].reshape(1,100,100) 
        expansion_img3 = self.expand_img3[index].reshape(1,100,100) 

        origin_img=torch.tensor(one_img,dtype=torch.float)
        expansion_img1=torch.tensor(expansion_img1,dtype=torch.float)
        expansion_img2=torch.tensor(expansion_img2,dtype=torch.float)
        expansion_img3=torch.tensor(expansion_img3,dtype=torch.float)
        
        img=torch.cat([expansion_img1, expansion_img2, expansion_img3, origin_img],dim=0)
        target=torch.tensor(self.label[index],dtype=torch.float)
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        img = self.normalizing(img)
        
        return img, target, index




    
class SoundFourValidInstance(Dataset):
    # Multi + Expansion Channel Dataset 10m x 10m, 1m x 1m resolution
    def __init__(self, path, index_set, expansion_list, t_mean=None, t_std = None,atype='total',transform=None,target_transform=None):

        self.path = path
        self.transform=transform
        self.target_transform=target_transform
        
        self.train_img10=np.load(path+'/train_img10.npy')
        self.train_label=np.load(path+'/train_label.npy')
        self.train_coord=np.load(path+'/train_coord.npy')    
        self.test_img10=np.load(path+'/test_img10.npy')
        self.test_label=np.load(path+'/test_label.npy')
        self.test_coord=np.load(path+'/test_coord.npy')    
        
        
        self.total_img = np.concatenate([self.train_img10,self.test_img10],axis=0)
        del self.train_img10
        del self.test_img10

        self.total_label = np.concatenate([self.train_label,self.test_label],axis=0)
        self.total_coord = np.concatenate([self.train_coord,self.test_coord],axis=0)

        del self.train_label
        del self.train_coord
        del self.test_label
        del self.test_coord

        self.total_expand1 = np.load(path+'/total_img{}.npy'.format(expansion_list[0]))
        self.total_expand2 = np.load(path+'/total_img{}.npy'.format(expansion_list[1]))
        self.total_expand3 = np.load(path+'/total_img{}.npy'.format(expansion_list[2]))

        self.expand_img1, self.expand_img2,self.expand_img3, self.origin_img, self.label, self.coord = parsing_index_four(self.total_expand1, self.total_expand2, 
                                                                                                                          self.total_expand3,self.total_img, 
                                                                                                                          self.total_label, self.total_coord, index_set)
        
        self.expand_img1 = self.expand_img1/255.0
        self.expand_img2 = self.expand_img2/255.0
        self.expand_img3 = self.expand_img3/255.0
        
        if (t_mean is None) & (t_std is None):
            self.mean_value, self.std_value = self.get_mean_var()
        else:
            self.mean_value = t_mean
            self.std_value = t_std

        self.expand_img1, self.expand_img2, self.expand_img3, self.origin_img, self.label= self.dataset_split(self.expand_img1,self.expand_img2,self.expand_img3,
                                                                            self.origin_img, self.label, atype=atype)
        
            
    def __len__(self):
        return self.origin_img.shape[0]

    def dataset_split(self,eimg1, eimg2, eimg3, oimg, label, atype):
        if atype =='total':
            return eimg1,eimg2,eimg3, oimg, label 
        
        elif atype =='center':
            index = np.load(self.path+'/center_idx.npy')

            eimg1 = eimg1[index]
            eimg2 = eimg2[index]
            eimg3 = eimg3[index]
            oimg = oimg[index]
            label = label[index]
            return eimg1,eimg2,eimg3, oimg, label
            
        elif atype == 'noncenter':
            index = np.load(self.path+'/noncenter_idx.npy')

            eimg1 = eimg1[index]
            eimg2 = eimg2[index]
            eimg3 = eimg3[index]
            oimg = oimg[index]
            label = label[index]
            return eimg1,eimg2,eimg3, oimg, label
        else:
            raise NotImplementedError()

    def get_mean_var(self):
        # 3 channel 
        
        mean_value0 = np.mean(np.mean(self.expand_img1,axis=(1,2)))
        std_value0 = np.mean(np.std(self.expand_img1,axis=(1,2)))
        
        mean_value1 = np.mean(np.mean(self.expand_img2,axis=(1,2)))
        std_value1 = np.mean(np.std(self.expand_img2,axis=(1,2)))
        
        mean_value2 = np.mean(np.mean(self.expand_img3,axis=(1,2)))
        std_value2 = np.mean(np.std(self.expand_img3,axis=(1,2)))
        
        mean_value3 = np.mean(np.mean(self.origin_img,axis=(1,2)))
        std_value3 = np.mean(np.std(self.origin_img,axis=(1,2)))

        mean_vec = torch.tensor((mean_value0,mean_value1,mean_value2, mean_value3),dtype=torch.float).unsqueeze(1).unsqueeze(1)
        std_vec = torch.tensor((std_value0,std_value1,std_value2, std_value3),dtype=torch.float).unsqueeze(1).unsqueeze(1)

        return mean_vec, std_vec

    def normalizing(self, x):
        return (x - self.mean_value)/self.std_value


    def __getitem__(self, index):
        one_img = self.origin_img[index].reshape(1,100,100)
        expansion_img1 = self.expand_img1[index].reshape(1,100,100) 
        expansion_img2 = self.expand_img2[index].reshape(1,100,100) 
        expansion_img3 = self.expand_img3[index].reshape(1,100,100) 

        origin_img=torch.tensor(one_img,dtype=torch.float)
        expansion_img1=torch.tensor(expansion_img1,dtype=torch.float)
        expansion_img2=torch.tensor(expansion_img2,dtype=torch.float)
        expansion_img3=torch.tensor(expansion_img3,dtype=torch.float)
        
        img=torch.cat([expansion_img1, expansion_img2, expansion_img3, origin_img],dim=0)
        target=torch.tensor(self.label[index],dtype=torch.float)
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        img = self.normalizing(img)
        
        return img, target, index
    
class SoundThreeValidInstance(Dataset):
    # Multi Channel Dataset 10m x 10m resolution
    def __init__(self, path, index_set, expansion_list, t_mean=None, t_std = None, atype='total',transform=None,target_transform=None):

        self.path = path
        self.transform=transform
        self.target_transform=target_transform
        
        self.train_img10=np.load(path+'/train_img10.npy')
        self.train_label=np.load(path+'/train_label.npy')
        self.train_coord=np.load(path+'/train_coord.npy')    
        self.test_img10=np.load(path+'/test_img10.npy')
        self.test_label=np.load(path+'/test_label.npy')
        self.test_coord=np.load(path+'/test_coord.npy')    
        
        
        self.total_img = np.concatenate([self.train_img10,self.test_img10],axis=0)
        del self.train_img10
        del self.test_img10

        self.total_label = np.concatenate([self.train_label,self.test_label],axis=0)
        self.total_coord = np.concatenate([self.train_coord,self.test_coord],axis=0)

        del self.train_label
        del self.train_coord
        del self.test_label
        del self.test_coord

        self.total_expand1 = np.load(path+'/total_img{}.npy'.format(expansion_list[0]))
        self.total_expand2 = np.load(path+'/total_img{}.npy'.format(expansion_list[1]))

        self.expand_img1, self.expand_img2, self.origin_img, self.label, self.coord = parsing_index_multi(self.total_expand1, self.total_expand2, 
                                                                                                                          self.total_img, self.total_label, 
                                                                                                                          self.total_coord, index_set)
        
        self.expand_img1 = self.expand_img1/255.0
        self.expand_img2 = self.expand_img2/255.0
        
        if (t_mean is None) & (t_std is None):
            self.mean_value, self.std_value = self.get_mean_var()
        else:
            self.mean_value = t_mean
            self.std_value = t_std

        self.expand_img1, self.expand_img2, self.origin_img, self.label= self.dataset_split(self.expand_img1,self.expand_img2,
                                                                            self.origin_img, self.label, atype=atype)
        
    def dataset_split(self,eimg1, eimg2, oimg, label, atype):
        if atype =='total':
            return eimg1,eimg2, oimg, label 
        
        elif atype =='center':
            index = np.load(self.path+'/center_idx.npy')

            eimg1 = eimg1[index]
            eimg2 = eimg2[index]
            oimg = oimg[index]
            label = label[index]
            return eimg1,eimg2, oimg, label
            
        elif atype == 'noncenter':
            index = np.load(self.path+'/noncenter_idx.npy')

            eimg1 = eimg1[index]
            eimg2 = eimg2[index]
            oimg = oimg[index]
            label = label[index]
            return eimg1,eimg2, oimg, label
        else:
            raise NotImplementedError()
    
    def __len__(self):
        return self.origin_img.shape[0]

    def get_mean_var(self):       
        mean_value1 = np.mean(np.mean(self.expand_img1,axis=(1,2)))
        std_value1 = np.mean(np.std(self.expand_img1,axis=(1,2)))
        
        mean_value2 = np.mean(np.mean(self.expand_img2,axis=(1,2)))
        std_value2 = np.mean(np.std(self.expand_img2,axis=(1,2)))
        
        mean_value3 = np.mean(np.mean(self.origin_img,axis=(1,2)))
        std_value3 = np.mean(np.std(self.origin_img,axis=(1,2)))

        mean_vec = torch.tensor((mean_value1,mean_value2, mean_value3),dtype=torch.float).unsqueeze(1).unsqueeze(1)
        std_vec = torch.tensor((std_value1,std_value2, std_value3),dtype=torch.float).unsqueeze(1).unsqueeze(1)
        return mean_vec, std_vec

    def normalizing(self, x):
        return (x - self.mean_value)/self.std_value


    def __getitem__(self, index):
        one_img = self.origin_img[index].reshape(1,100,100)
        expansion_img1 = self.expand_img1[index].reshape(1,100,100) 
        expansion_img2 = self.expand_img2[index].reshape(1,100,100) 

        origin_img=torch.tensor(one_img,dtype=torch.float)
        expansion_img1=torch.tensor(expansion_img1,dtype=torch.float)
        expansion_img2=torch.tensor(expansion_img2,dtype=torch.float)
        
        img=torch.cat([expansion_img1, expansion_img2, origin_img],dim=0)
        target=torch.tensor(self.label[index],dtype=torch.float)


        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        img = self.normalizing(img)
        
        return img, target, index


class SoundSingleValidInstance(Dataset):
    # Single Channel Dataset 10m x 10m resolution
    def __init__(self, path, index_set, t_mean=None, t_std = None, atype='total',transform=None,target_transform=None):

        self.transform=transform
        self.target_transform=target_transform

        self.train_img10=np.load(path+'/train_img10.npy')
        self.train_label=np.load(path+'/train_label.npy')
        self.train_coord=np.load(path+'/train_coord.npy')    
        self.test_img10=np.load(path+'/test_img10.npy')
        self.test_label=np.load(path+'/test_label.npy')
        self.test_coord=np.load(path+'/test_coord.npy')    
        

        self.total_img10 = np.concatenate([self.train_img10,self.test_img10],axis=0)
        del self.train_img10
        del self.test_img10

        self.total_label = np.concatenate([self.train_label,self.test_label],axis=0)
        self.total_coord = np.concatenate([self.train_coord,self.test_coord],axis=0)

        del self.train_label
        del self.train_coord
        del self.test_label
        del self.test_coord

        self.origin_img, self.label, self.coord = parsing_index(self.total_img10, 
                                                                    self.total_label, 
                                                                    self.total_coord, 
                                                                    index_set 
                                                                    )
        
        
        if (t_mean is None) & (t_std is None):
            self.mean_value, self.std_value = self.get_mean_var()
        else:
            self.mean_value = t_mean
            self.std_value = t_std
            
        self.origin_img, self.label= self.dataset_split(self.origin_img, self.label, atype=atype)
        
    def dataset_split(self,oimg, label, atype):
        if atype =='total':
            return oimg, label 
        
        elif atype =='center':
            index = np.load(self.path+'/center_idx.npy')

            oimg = oimg[index]
            label = label[index]
            return oimg, label
            
        elif atype == 'noncenter':
            index = np.load(self.path+'/noncenter_idx.npy')

            oimg = oimg[index]
            label = label[index]
            return oimg, label
        else:
            raise NotImplementedError()

    
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
        target=torch.tensor(self.label[index],dtype=torch.float)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        img = self.normalizing(img)

        return img, target, index



class SoundTwoValidInstance(Dataset):
    # Single + Expansion Channel Dataset 10m x 10m, 1m x 1m resolution
    def __init__(self, path, index_set, expansion, t_mean=None, t_std = None,atype='total',transform=None,target_transform=None):

        self.transform=transform
        self.target_transform=target_transform
        
        self.train_img10=np.load(path+'/train_img10.npy')
        self.train_label=np.load(path+'/train_label.npy')
        self.train_coord=np.load(path+'/train_coord.npy')    
        self.test_img10=np.load(path+'/test_img10.npy')
        self.test_label=np.load(path+'/test_label.npy')
        self.test_coord=np.load(path+'/test_coord.npy')    
        self.total_expand = np.load(path+'./total_img{}.npy'.format(expansion))

        self.total_img10 = np.concatenate([self.train_img10,self.test_img10],axis=0)

        del self.train_img10
        del self.test_img10

        self.total_label = np.concatenate([self.train_label,self.test_label],axis=0)
        self.total_coord = np.concatenate([self.train_coord,self.test_coord],axis=0)

        del self.train_label
        del self.train_coord
        del self.test_label
        del self.test_coord

        self.expansion_img, self.origin_img, self.label, self.coord = parsing_index(self.total_img10, 
                                                                                    self.total_label, 
                                                                                    self.total_coord, 
                                                                                    index_set, 
                                                                                    self.total_expand)
        self.expansion_img = self.expansion_img/255.0
        
        if (t_mean is None) & (t_std is None):
            self.mean_value, self.std_value = self.get_mean_var()
        else:
            self.mean_value = t_mean
            self.std_value = t_std

        self.expansion_img, self.origin_img, self.label= self.dataset_split(self.expansion_img, self.origin_img, self.label, atype=atype)
        
            
    def __len__(self):
        return self.origin_img.shape[0]

    def dataset_split(self,eimg, oimg, label, atype):
        if atype =='total':
            return eimg, oimg, label 
        
        elif atype =='center':
            index = np.load(self.path+'/center_idx.npy')

            eimg = eimg[index]
            oimg = oimg[index]
            label = label[index]
            return eimg, oimg, label
            
        elif atype == 'noncenter':
            index = np.load(self.path+'/noncenter_idx.npy')

            eimg = eimg[index]
            oimg = oimg[index]
            label = label[index]
            return eimg, oimg, label
        else:
            raise NotImplementedError()

    def get_mean_var(self):
        # 3 channel         
        mean_value0 = np.mean(np.mean(self.expansion_img,axis=(1,2)))
        std_value0 = np.mean(np.std(self.expansion_img,axis=(1,2)))
        
        mean_value1 = np.mean(np.mean(self.origin_img,axis=(1,2)))
        std_value1 = np.mean(np.std(self.origin_img,axis=(1,2)))

        mean_vec = torch.tensor((mean_value0,mean_value1),dtype=torch.float).unsqueeze(1).unsqueeze(1)
        std_vec = torch.tensor((std_value0,std_value1),dtype=torch.float).unsqueeze(1).unsqueeze(1)

        return mean_vec, std_vec

    def normalizing(self, x):
        return (x - self.mean_value)/self.std_value


    def __getitem__(self, index):
        one_img = self.origin_img[index].reshape(1,100,100)
        expansion_img = self.expansion_img[index].reshape(1,100,100) 

        origin_img=torch.tensor(one_img,dtype=torch.float)
        expansion_img=torch.tensor(expansion_img,dtype=torch.float)
        target=torch.tensor(self.label[index],dtype=torch.float)

        img = torch.cat([expansion_img, origin_img],dim=0)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        img = self.normalizing(img)
        
        return img, target, index




def get_single_dataloaders(path,batch_size=128, num_workers=4, seed=0):
    
    """
    Sound Map data efficient aug Experiment for single channel dataset
    """
    train_idx, valid_idx = choose_region(seed)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        #transforms.Normalize((0.90865), (0.21802)),
    ])
    
    # test_transform = transforms.Compose([
    #     transforms.Normalize((0.90865), (0.21802)),
    # ])


    train_set = SoundInstance(path, train_idx, transform = train_transform)
    n_data = len(train_set)
    
    t_mean = train_set.mean_value
    t_std = train_set.std_value

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = SoundInstance(path, valid_idx, t_mean=t_mean, t_std = t_std)

    test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=num_workers)


    return train_loader, test_loader, n_data


def get_two_dataloaders(path,resolution,batch_size=128, num_workers=4, seed=0):
    
    """
    Sound Map multi + expansion data
    """
    train_idx, valid_idx = choose_region(seed)
    train_transform = transforms.Compose([
        SoundRandomHorizontalFlip(),
        SoundRandomVerticalFlip(),
        # transforms.Normalize((0.92967, 0.91521, 0.99345 ,0.90865), (0.13998, 0.21451, 0.04454 ,0.21802)),
    ])
        
    # test_transform = transforms.Compose([
    #     transforms.Normalize((0.92967, 0.91521, 0.99345 ,0.90865), (0.13998, 0.21451, 0.04454 ,0.21802)),
    # ])


    train_set = SoundTwoChannelInstance(path,train_idx, resolution, transform = train_transform)
    n_data = len(train_set)
    
    t_mean = train_set.mean_value
    t_std = train_set.std_value

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = SoundTwoChannelInstance(path,valid_idx, t_mean=t_mean, t_std = t_std)

    test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=num_workers)


    return train_loader, test_loader, n_data


def get_three_sound_dataloaders(path, resolution_list, batch_size=128, num_workers=4, seed=0):
    
    """
    Sound Map multi channel data
    """
    train_idx, valid_idx = choose_region(seed)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    #     transforms.Normalize((0.91521, 0.99345 ,0.90865), (0.21451, 0.04454 ,0.21802)),
    ])
    
    # test_transform = transforms.Compose([
    #     transforms.Normalize((0.91521, 0.99345 ,0.90865), (0.21451, 0.04454 ,0.21802)),
    # ])


    train_set = SoundThreeChannelInstance(path,train_idx, resolution_list, transform = train_transform)
    n_data = len(train_set)
    
    t_mean = train_set.mean_value
    t_std = train_set.std_value

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = SoundThreeChannelInstance(path, valid_idx, resolution_list, t_mean = t_mean, t_std = t_std)

    test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=num_workers)


    return train_loader, test_loader, n_data

def get_four_sound_dataloaders(path, resolution_list, batch_size=128, num_workers=4, seed=0):
    
    """
    Sound Map multi + expansion data
    """
    train_idx, valid_idx = choose_region(seed)
    train_transform = transforms.Compose([
        SoundRandomHorizontalFlip(),
        SoundRandomVerticalFlip(),
        # transforms.Normalize((0.92967, 0.91521, 0.99345 ,0.90865), (0.13998, 0.21451, 0.04454 ,0.21802)),
    ])
        
    # test_transform = transforms.Compose([
    #     transforms.Normalize((0.92967, 0.91521, 0.99345 ,0.90865), (0.13998, 0.21451, 0.04454 ,0.21802)),
    # ])


    train_set = SoundFourChannelInstance(path, train_idx, resolution_list, transform = train_transform)
    n_data = len(train_set)
    
    t_mean = train_set.mean_value
    t_std = train_set.std_value

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = SoundFourChannelInstance(path, valid_idx, resolution_list, t_mean=t_mean, t_std = t_std)

    test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=num_workers)


    return train_loader, test_loader, n_data




def get_single_valid_dataloaders(path,batch_size=128, num_workers=4, seed=0,atype='total'):
    
    """
    Sound Map data efficient aug Experiment for single channel dataset
    """
    train_idx, valid_idx = choose_region(seed)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        #transforms.Normalize((0.90865), (0.21802)),
    ])
    
    # test_transform = transforms.Compose([
    #     transforms.Normalize((0.90865), (0.21802)),
    # ])


    train_set = SoundSingleValidInstance(path, train_idx, transform = train_transform)
    n_data = len(train_set)
    
    t_mean = train_set.mean_value
    t_std = train_set.std_value

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = SoundSingleValidInstance(path, valid_idx, t_mean=t_mean, t_std = t_std, atype=atype)

    test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=num_workers)


    return train_loader, test_loader, n_data

def get_two_valid_dataloaders(path, resolution, batch_size=128, num_workers=4, seed=0,atype='total'):
    
    """
    Sound Map Base + expansion data
    """
    train_idx, valid_idx = choose_region(seed)
    train_transform = transforms.Compose([
        SoundRandomHorizontalFlip(),
        SoundRandomVerticalFlip(),
        # transforms.Normalize((0.92967, 0.91521, 0.99345 ,0.90865), (0.13998, 0.21451, 0.04454 ,0.21802)),
    ])
        
    # test_transform = transforms.Compose([
    #     transforms.Normalize((0.92967, 0.91521, 0.99345 ,0.90865), (0.13998, 0.21451, 0.04454 ,0.21802)),
    # ])


    train_set = SoundTwoValidInstance(path,train_idx, resolution, transform = train_transform)
    n_data = len(train_set)
    
    t_mean = train_set.mean_value
    t_std = train_set.std_value

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = SoundTwoValidInstance(path,valid_idx, resolution, t_mean=t_mean, t_std = t_std, atype=atype)

    test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=num_workers)


    return train_loader, test_loader, n_data



def get_four_valid_dataloaders(path,resolution_list,batch_size=128, num_workers=4, seed=0,atype='total'):
    
    """
    Sound Map multi + expansion data
    """
    train_idx, valid_idx = choose_region(seed)
    train_transform = transforms.Compose([
        SoundRandomHorizontalFlip(),
        SoundRandomVerticalFlip(),
        # transforms.Normalize((0.92967, 0.91521, 0.99345 ,0.90865), (0.13998, 0.21451, 0.04454 ,0.21802)),
    ])
        

    train_set = SoundFourValidInstance(path,train_idx,resolution_list, transform = train_transform)
    n_data = len(train_set)
    
    t_mean = train_set.mean_value
    t_std = train_set.std_value

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = SoundFourValidInstance(path,valid_idx, resolution_list, t_mean=t_mean, t_std = t_std, atype=atype)

    test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=num_workers)


    return train_loader, test_loader, n_data


def get_three_valid_dataloaders(path,resolution_list, batch_size=128, num_workers=4, seed=0,atype='total'):
    
    """
    Sound Map multi channel data
    """
    train_idx, valid_idx = choose_region(seed)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    #     transforms.Normalize((0.91521, 0.99345 ,0.90865), (0.21451, 0.04454 ,0.21802)),
    ])
    
    # test_transform = transforms.Compose([
    #     transforms.Normalize((0.91521, 0.99345 ,0.90865), (0.21451, 0.04454 ,0.21802)),
    # ])


    train_set = SoundThreeValidInstance(path,train_idx,resolution_list, transform = train_transform)
    n_data = len(train_set)
    
    t_mean = train_set.mean_value
    t_std = train_set.std_value

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = SoundThreeValidInstance(path, valid_idx,resolution_list, t_mean = t_mean, t_std = t_std, atype=atype)

    test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=num_workers)


    return train_loader, test_loader, n_data



