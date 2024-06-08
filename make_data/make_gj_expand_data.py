import torch
import numpy as np
from utils.util import *
from tqdm import tqdm
import torch.nn.functional as F
import rasterio
import argparse


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--resolution', type=int, default=2, help='pixel resolution')
    parser.add_argument('--region', type=str, default='GJ', help='region')
    
    parser.add_argument('--flag', type=int, default=1, help='Whether you store data')
    opt = parser.parse_args()
    return opt


def making_pixel(build, wall, road):
    # Rule of Building pixel : Case - There are building value and wall value at the same time, select maximum value.
    b_pixel = np.maximum(build,wall)
    
    # Case - There are building pixel value (not 0) and road pixel value, allocate build pixel.
    # Case - If one of these pixel values is zero, allocate non zero pixel value.
    # Case - If both pixel values is zero, allocate zero pixel value.
    pixel =np.where((b_pixel !=0) & (road !=0), b_pixel, np.maximum(b_pixel, road))
    
    return pixel

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
    opt = parse_option()    
    file_path = './assets/0401_data/GJ/'
    tcord= np.load('./assets/newdata/train_coord.npy')
    vcord= np.load('./assets/newdata/test_coord.npy')
    coord = np.concatenate([tcord, vcord], axis=0)
    real_coord = (coord * 10) + np.array([[167120,272260]])
    
    del tcord
    del vcord
    
    # Fetch 1m resolution Tif file.
    
    build_dataset = rasterio.open(file_path + 'building_1m.tif')
    road_dataset = rasterio.open(file_path + 'Road_1m_D.tif')
    wall_dataset = rasterio.open(file_path + 'Wall_1m.tif')
    
    build = build_dataset.read(1).reshape(1,27740,45330)
    road = road_dataset.read(1).reshape(1,27740,45330)
    wall = wall_dataset.read(1).reshape(1,27740,45330)
    
    pixel = making_pixel(build,wall,road) # 1 x 1 resolution -> (27740, 45330)
    pixel= pixel.reshape(27740,45330)
    
    del build
    del road
    del wall
    
    index_list = []
    for i in range(real_coord.shape[0]):
        index_list.append(build_dataset.index(real_coord[i][0],real_coord[i][1]))

    lu_direction = int(100 * opt.resolution // 2)
    rl_direction = int(100 * opt.resolution // 2 - 1)
    
    img = []
    if opt.resolution > 1 :

        for i in tqdm(range(real_coord.shape[0])):
            temp_cord = make_grid(index_list[i],lu_direction,rl_direction)
            temp_img = pixel[temp_cord[:,0],temp_cord[:,1]].reshape(100*opt.resolution,100*opt.resolution,order='F')
            temp_img = torch.FloatTensor(temp_img)            
            temp_img = F.max_pool2d(temp_img.unsqueeze(0), opt.resolution).squeeze(0).numpy()
            temp_img = temp_img.astype('uint8') 
            # Allocate 255 if pixel value is 0.
            temp_img = np.where(temp_img == 0 ,255, temp_img)
            img.append(temp_img)          
            
            
    else:
        for i in tqdm(range(real_coord.shape[0])):
            temp_cord = make_grid(index_list[i],lu_direction,rl_direction)
            temp_img = pixel[temp_cord[:,0],temp_cord[:,1]].reshape(100*opt.resolution,100*opt.resolution,order='F')
            temp_img = temp_img.astype('uint8')
            # Allocate 255 if pixel value is 0.
            temp_img = np.where(temp_img == 0 ,255, temp_img)
            img.append(temp_img)
            
    img = np.array(img) # img shape : B, 100, 100
    
    if opt.flag == 1:
        np.save('./assets/newdata/total_img{}'.format(opt.resolution),img)
    
    else:
        print('Not Save') 

    print(opt.region, " : Handling Success")

if __name__ == '__main__':
    main()
    
    
