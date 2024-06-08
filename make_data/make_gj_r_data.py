import torch
import numpy as np
from utils.util import *
from tqdm import tqdm
import torch.nn.functional as F
import rasterio
import argparse

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
def main():
    file_path = './assets/newdata/'
    tcord= np.load('./assets/newdata/train_coord.npy')
    vcord= np.load('./assets/newdata/test_coord.npy')
    coord = np.concatenate([tcord, vcord], axis=0)
    real_coord = (coord * 10) + np.array([[167120,272260]])
        
    del tcord
    del vcord

    # Fetch 1m resolution Tif file.
    car_dataset = rasterio.open(file_path + 'Car.tif')
    road_speed_dataset = rasterio.open(file_path + 'Road_Speed.tif')
    truck_dataset = rasterio.open(file_path + 'Truck.tif')
    wall_dataset = rasterio.open(file_path + 'Wall_area.tif')

    car = car_dataset.read(1).reshape(1,5548,9066)
    truck = truck_dataset.read(1).reshape(1,5548,9066)
    road = road_speed_dataset.read(1).reshape(1,5548,9066)
    wall = wall_dataset.read(1).reshape(1,5548,9066)

    pixel = np.concatenate([car,truck,road,wall],axis=0)

    index_list = []
    for i in range(coord.shape[0]):
        index_list.append(car_dataset.index(real_coord[i][0],real_coord[i][1]))

    lu_direction = int(100)
    rl_direction = int(100-1)

    img = np.zeros((258730,4,100,100))

    for i in tqdm(range(real_coord.shape[0])):
        temp_cord = make_grid(index_list[i],lu_direction,rl_direction)
        temp_img = pixel[:,temp_cord[:,0],temp_cord[:,1]].reshape(4,100*2,100*2,order='F')
        temp_img = torch.FloatTensor(temp_img)            
        temp_img = F.max_pool2d(temp_img.unsqueeze(0), 2).squeeze(0).numpy()

        img[i] = temp_img          
                    
    img = np.array(img) # img shape : B, 100, 100

    np.save('./assets/newdata/total_r_img',img)

if __name__=="__main__":
    main()
    
