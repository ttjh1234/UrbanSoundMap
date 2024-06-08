import argparse
import numpy as np
import pandas as pd
import mat73
import scipy.io
import h5py
import torch
import random

def parse_option():

    parser = argparse.ArgumentParser('argument for generating data')
    parser.add_argument('--mat_file_path', type=str, default = "./assets/testregion/",help='Mat File Path')
    parser.add_argument('--region', type=str, default = 'DJ', choices= ['DJ','Seoul'], help = 'Region')
    parser.add_argument('--save_flag', type=int, default = 0,help='File Save Flag 0 is no save, if you want to save, use 1' )
    opt = parser.parse_args()
    
    return opt

def check_negative_label(img1, img10, coord, label, dataset = 'Train'):
    print("{} negative label index : ".format(dataset), np.where(label<=0)[0])

    if np.where(label<=0)[0].shape[0]!=0:
        img1 = np.delete(img1,np.where(label<=0)[0],axis=0)
        img10 = np.delete(img10,np.where(label<=0)[0],axis=0)
        coord = np.delete(coord, np.where(label<=0)[0], axis=0)
        label = np.delete(label, np.where(label<=0)[0], axis=0)
        return img1, img10, coord, label
    else:
        return img1, img10, coord, label
    
def check_over_label(img1,img10,label,coord):
    over_index = np.where(label==1000)[0]

    img1 = np.delete(img1, over_index, axis=0)
    img10 = np.delete(img10, over_index, axis=0)
    label = np.delete(label, over_index, axis=0)
    coord = np.delete(coord, over_index, axis=0)

    return img1, img10, label, coord
    
def read_matfile_and_split(mat_file_path):
    '''
    Matfile includes each dataset's 1mx1m image, 10mx10m image, coordinate, label
    '''
    mat_file = mat73.loadmat(mat_file_path)    

    total_img10 = mat_file['database_10m'].astype('uint8')
    total_img1 = mat_file['database_1m'].astype('uint8')
    total_cord = mat_file['cord']
    total_label = mat_file['datalabel']

    total_img10 = total_img10.transpose(2,0,1)
    total_img1 = total_img1.transpose(2,0,1)
    total_img10 = np.where(total_img10==0,255,total_img10)
    total_img1 = np.where(total_img1==0,255,total_img1)
 
    total_img1, total_img10, total_label, total_cord =check_over_label(total_img1, total_img10, total_label, total_cord)
 
    return total_img1, total_img10, total_cord, total_label


def main():
    
    opt = parse_option()
    
    file_path = opt.mat_file_path + opt.region + '_whole_city.mat'
    
    test_img1, test_img10, test_cord, test_label = read_matfile_and_split(file_path)
    
    hdf5_file = './assets/newdata/test_region_{}.h5py'.format(opt.region)
    
    if opt.save_flag==1:
        with h5py.File(hdf5_file, 'w') as hf:
            hf.create_dataset('test_img10',data=test_img10,chunks=(10000, 100, 100))
            hf.create_dataset('test_img1',data=test_img1, chunks=(10000,100,100))
            hf.create_dataset('test_label',data=test_label, chunks=(20000,))
            hf.create_dataset('test_coord',data=test_cord, chunks=(20000,2))
        
    print('Data Generate Complete')
    
if __name__ == '__main__':
    main()