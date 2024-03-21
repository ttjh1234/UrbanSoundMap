import torch
import numpy as np
import os
import pandas as pd
import mat73
import scipy.io
from utils.util import *
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import rasterio
import random
import h5py
import argparse


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--resolution', type=int, default=1000, help='resolution')
    parser.add_argument('--pool', type=int, default=2, help='pool')

    opt = parser.parse_args()
    
    return opt

def make_grid(x, half_side,half_side2=None):
    if half_side2 is None:
        ref_x,ref_y = x
        x_min = ref_x - half_side
        x_max = ref_x + half_side
        y_min = ref_y - half_side
        y_max = ref_y + half_side
        candidate_x = np.arange(x_min, x_max+1, step=1)
        candidate_y = np.arange(y_min, y_max+1, step=1)

        cand_x, cand_y = np.meshgrid(candidate_x, candidate_y)
        cand_x = cand_x.ravel().reshape(-1,1)
        cand_y = cand_y.ravel().reshape(-1,1)

        cand_index = np.concatenate([cand_x,cand_y],axis=1)
        #cand_index = np.delete(cand_index,((half_side*2+1)**2-1)//2,axis=0)
        
        return cand_index.astype(int)
    else:
        ref_x,ref_y = x
        x_min = ref_x - half_side
        x_max = ref_x + half_side2
        y_min = ref_y - half_side
        y_max = ref_y + half_side2
        candidate_x = np.arange(x_min, x_max+1, step=1)
        candidate_y = np.arange(y_min, y_max+1, step=1)

        cand_x, cand_y = np.meshgrid(candidate_x, candidate_y)
        cand_x = cand_x.ravel().reshape(-1,1)
        cand_y = cand_y.ravel().reshape(-1,1)

        cand_index = np.concatenate([cand_x,cand_y],axis=1)
        #cand_index = np.delete(cand_index,((half_side*2+1)**2-1)//2,axis=0)
        
        return cand_index.astype(int)

def make_representation_pooling(data, npool):
    # data : 이웃 개수 * 7차원.
    hw = data.shape[0]
    h = int(np.sqrt(hw))
    data_re = data.reshape(h,h,-1).transpose(2,0,1)
    represent = F.avg_pool2d(torch.tensor(data_re), (npool,npool)) / (npool)
    represent = represent.view(-1).numpy()

    return represent


def main():
    opt = parse_option()
    
    file_path = './assets/newdata/'
    tcord= np.load(file_path+'train_coord.npy')
    vcord= np.load(file_path+'test_coord.npy')

    cord = np.concatenate([tcord, vcord],axis=0)
    real_coord = (cord * 10) + np.array([[167120,272260]])

    car_dataset = rasterio.open(file_path + 'Car.tif')
    FSI_dataset = rasterio.open(file_path + 'FSI.tif')
    GSI_dataset = rasterio.open(file_path + 'GSI.tif')
    Road_Area_dataset = rasterio.open(file_path + 'Road_Area.tif')
    Road_Speed_dataset = rasterio.open(file_path + 'Road_Speed.tif')
    Truck_dataset = rasterio.open(file_path + 'Truck.tif')
    Wall_area_dataset = rasterio.open(file_path + 'Wall_area.tif')

    index_list = []
    for i in range(real_coord.shape[0]):
        index_list.append(car_dataset.index(real_coord[i][0],real_coord[i][1]))

    ind = np.array(index_list)

    c1 = car_dataset.read(1).reshape(1,5548,9066)
    c2 = FSI_dataset.read(1).reshape(1,5548,9066)
    c3 = GSI_dataset.read(1).reshape(1,5548,9066)
    c4 = Road_Area_dataset.read(1).reshape(1,5548,9066)
    c5 = Road_Speed_dataset.read(1).reshape(1,5548,9066)
    c6 = Truck_dataset.read(1).reshape(1,5548,9066)
    c7 = Wall_area_dataset.read(1).reshape(1,5548,9066)

    data = np.concatenate([c1,c2,c3,c4,c5,c6,c7],axis=0)

    data = data.transpose(1,2,0)

    coord = pd.DataFrame(ind)

    variable = []
    for i in range(coord.shape[0]):
        temp_data = make_grid(coord.iloc[i],(opt.resolution//5)-1,(opt.resolution//5))
        x_mask = (temp_data[:,0]>=5548) | (temp_data[:,0]<0)
        y_mask = (temp_data[:,1]>=9066) | (temp_data[:,1]<0)
        if np.where(x_mask | y_mask)[0].shape[0] !=0 :
            continue       
        variable.append(make_representation_pooling(data[temp_data[:,0],temp_data[:,1],:],opt.pool))
    rep_var= np.array(variable)
    print(rep_var.shape)
    np.save('./assets/newdata/urbanform{}pool{}'.format(opt.resolution, opt.pool),rep_var)

if __name__ == '__main__':
    main()
