'''
This file contains code to mask train & valid dataset.

Transform Mat file to python Numpy file.
Drop anomaly data and duplicated data.
Finally, Normalizing pixels in [0,1]

'''

import numpy as np
import argparse

def parse_option():

    parser = argparse.ArgumentParser('argument for generating data')
    parser.add_argument('--save_path', type=str, default = "./assets/newdata/",help='File Save Path')
    parser.add_argument('--save_flag', type=int, default = 0,help='File Save Flag 0 is no save, if you want to save, use 1' )
    opt = parser.parse_args()
    
    return opt


def main():

    opt = parse_option()
    
    train_img1 = np.load(opt.save_path+'train_img1.npy')
    train_img10 = np.load(opt.save_path+'train_img10.npy')
    train_label = np.load(opt.save_path+'train_label.npy')
    train_coord = np.load(opt.save_path+'train_coord.npy')
    
    test_img1 = np.load(opt.save_path+'test_img1.npy')
    test_img10 = np.load(opt.save_path+'test_img10.npy')
    test_label = np.load(opt.save_path+'test_label.npy')
    test_coord = np.load(opt.save_path+'test_coord.npy')


    total_img1 = np.concatenate([train_img1, test_img1],axis=0)
    total_img10 = np.concatenate([train_img10, test_img10],axis=0)
    total_label = np.concatenate([train_label, test_label],axis=0)
    total_coord = np.concatenate([train_coord, test_coord],axis=0)
    
    if opt.save_flag==1:
        np.save(opt.save_path+'total_img1',total_img1)
        np.save(opt.save_path+'total_img10',total_img10)
        np.save(opt.save_path+'total_label',total_label)
        np.save(opt.save_path+'total_coord',total_coord)

    print(total_img1.shape)
    print(total_img10.shape)
    print(total_label.shape)
    print(total_coord.shape)


if __name__ == '__main__':
    main()