'''
This file contains code to mask train & valid dataset.

Transform Mat file to python Numpy file.
Drop anomaly data and duplicated data.
Finally, Normalizing pixels in [0,1]

'''
import argparse
import numpy as np
import pandas as pd
import mat73
import scipy.io

def parse_option():

    parser = argparse.ArgumentParser('argument for generating data')
    parser.add_argument('--mat_file_path', type=str, default = "./assets/source_data/GJ_NEW_SAMPLING_1m_10m_10P_mod.mat",help='Mat File Path')
    parser.add_argument('--save_path', type=str, default = "./assets/newdata/",help='File Save Path')
    parser.add_argument('--save_flag', type=int, default = 0,help='File Save Flag 0 is no save, if you want to save, use 1' )
    opt = parser.parse_args()
    
    return opt


def read_matfile_and_split(mat_file_path):
    '''
    Matfile includes each dataset's 1mx1m image, 10mx10m image, coordinate, label
    '''
    mat_file = mat73.loadmat(mat_file_path)    

    train_img1 = mat_file['train_database_1m']
    train_img10 = mat_file['train_database_10m']
    train_cord = mat_file['train_cord']
    train_label = mat_file['train_datalabel']

    val_img1 = mat_file['val_database_1m']
    val_img10 = mat_file['val_database_10m']
    val_cord = mat_file['val_cord']
    val_label = mat_file['val_datalabel']

    train_img1 = train_img1.transpose(2,0,1)
    train_img10 = train_img10.transpose(2,0,1)
    val_img1 = val_img1.transpose(2,0,1)
    val_img10 = val_img10.transpose(2,0,1)

    train_img1_re = np.where(train_img1==0,255,train_img1)
    val_img1_re = np.where(val_img1==0,255,val_img1)
    
    return train_img1_re, train_img10, train_cord, train_label, val_img1_re, val_img10, val_cord, val_label,

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
    
def check_duplicated(img1, img10, coord, label, opt, dataset = 'Train'):
    print("{} # of overlapped data: ".format(dataset), np.where(pd.DataFrame(coord).duplicated()==True)[0].shape)
    dup_index = np.where(pd.DataFrame(coord).duplicated()==True)[0]

    overlap_list = []
    diffn=0
    diff_list = []
    area = img1.shape[1] * img1.shape[2]
    for i in dup_index:
        temp_coord = coord[i]
        eq_sample_index = np.where((coord[:,0]==temp_coord[0]) & (coord[:,1]==temp_coord[1]))[0]
        if eq_sample_index.shape[0]!=2:
            print("Dup 2 over")

        if np.sum(img10[i] == img10[eq_sample_index[0]]) != area :
            print("oh 10m")
        if np.sum(img1[i] == img1[eq_sample_index[0]]) != area :
            print("oh 1m")
        
        if label[i] != label[eq_sample_index[0]] :
            print(label[i], label[eq_sample_index[0]])
            diff_list.append(label[i]-label[eq_sample_index[0]])
            diffn +=1
        
        overlap_list.append([eq_sample_index[0],i])

    # 
    overlap_index = pd.DataFrame(overlap_list,columns=['first','second'])
    if opt.save_flag==1:
        overlap_index.to_csv(opt.path+'{}_overlap.csv'.format(dataset.lower()),index=False)
    
    remove_ind = overlap_index['second'].values

    img1 = np.delete(img1, remove_ind, axis=0)
    img10 = np.delete(img10, remove_ind, axis=0)
    label = np.delete(label, remove_ind, axis=0)
    coord = np.delete(coord, remove_ind, axis=0)
    
    # Recheck
    print("## Recheck ##")
    print("{} # of overlapped data: ".format(dataset), np.where(pd.DataFrame(coord).duplicated()==True)[0].shape)
    
    if np.where(pd.DataFrame(coord).duplicated()==True)[0].shape[0] !=0:
        raise NotImplementedError()
    
    return img1, img10, label, coord

def normalize_pixel_value(img):
    return img / 255.0

def main():

    opt = parse_option()
    
    train_img1, train_img10, train_coord, train_label, val_img1, val_img10, val_coord, val_label = read_matfile_and_split(opt.mat_file_path)

    train_img1, train_img10, train_coord, train_label = check_negative_label(train_img1, train_img10, train_coord, train_label, dataset='Train')
    val_img1, val_img10, val_coord, val_label = check_negative_label(val_img1, val_img10, val_coord, val_label, dataset='Valid')

    train_img1, train_img10, train_coord, train_label = check_duplicated(train_img1, train_img10, train_coord, train_label, dataset='Train')
    val_img1, val_img10, val_coord, val_label = check_duplicated(val_img1, val_img10, val_coord, val_label, dataset='Valid')

    train_img1 = normalize_pixel_value(train_img1)
    train_img10 = normalize_pixel_value(train_img10)
    val_img1 = normalize_pixel_value(val_img1)
    val_img10 = normalize_pixel_value(val_img10)

    if opt.save_flag==1:
        np.save(opt.save_path+'train_img1',train_img1)
        np.save(opt.save_path+'train_img10',train_img10)
        np.save(opt.save_path+'train_label',train_label)
        np.save(opt.save_path+'train_coord',train_coord)

        np.save(opt.save_path+'test_img1',val_img1)
        np.save(opt.save_path+'test_img10',val_img10)
        np.save(opt.save_path+'test_label',val_label)
        np.save(opt.save_path+'test_coord',val_coord)

    print(train_img1.shape)
    print(train_img10.shape)
    print(train_label.shape)
    print(train_coord.shape)

    print(val_img1.shape)
    print(val_img10.shape)
    print(val_label.shape)
    print(val_coord.shape)



if __name__ == '__main__':
    main()