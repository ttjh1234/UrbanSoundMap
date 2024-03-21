import argparse
import numpy as np
import pandas as pd
import mat73
import scipy.io
import h5py


def parse_option():

    parser = argparse.ArgumentParser('argument for generating data')
    parser.add_argument('--mat_file_path', type=str, default = "./assets/GJ_whole_city.mat",help='Mat File Path')
    parser.add_argument('--save_path', type=str, default = "./assets/total_data/",help='File Save Path')
    parser.add_argument('--save_flag', type=int, default = 0,help='File Save Flag 0 is no save, if you want to save, use 1' )
    opt = parser.parse_args()
    
    return opt


def read_matfile_and_split(mat_file_path):
    '''
    Matfile includes each dataset's 1mx1m image, 10mx10m image, coordinate, label
    '''
    mat_file = mat73.loadmat(mat_file_path)    

    total_img10 = mat_file['database_10m']
    total_img1 = mat_file['database_1m'].astype('uint8')
    total_cord = mat_file['cord']
    total_label = mat_file['datalabel']

    total_img10 = total_img10.transpose(2,0,1)
    total_img1 = total_img1.transpose(2,0,1)
    total_img1 = np.where(total_img1==0,255,total_img1)
 
    return total_img1, total_img10, total_cord, total_label

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

    # overlap_list = []
    # diffn=0
    # diff_list = []
    # area = img10.shape[1] * img10.shape[2]
    # for i in dup_index:
    #     temp_coord = coord[i]
    #     eq_sample_index = np.where((coord[:,0]==temp_coord[0]) & (coord[:,1]==temp_coord[1]))[0]
    #     if eq_sample_index.shape[0]!=2:
    #         print("Dup 2 over")

    #     if np.sum(img10[i] == img10[eq_sample_index[0]]) != area :
    #         print("oh 10m")
        
    #     if label[i] != label[eq_sample_index[0]] :
    #         print(label[i], label[eq_sample_index[0]])
    #         diff_list.append(label[i]-label[eq_sample_index[0]])
    #         diffn +=1
        
    #     overlap_list.append([eq_sample_index,i])

    # # 
    # overlap_index = pd.DataFrame(overlap_list,columns=['first','second'])
    # if opt.save_flag==1:
    #     overlap_index.to_csv(opt.save_path+'{}_overlap.csv'.format(dataset.lower()),index=False)
    
    # remove_ind = overlap_index['first'].values

    img1 = np.delete(img1, dup_index, axis=0)
    img10 = np.delete(img10, dup_index, axis=0)
    label = np.delete(label, dup_index, axis=0)
    coord = np.delete(coord, dup_index, axis=0)
    
    # Recheck
    print("## Recheck ##")
    print("{} # of overlapped data: ".format(dataset), np.where(pd.DataFrame(coord).duplicated()==True)[0].shape)
    
    if np.where(pd.DataFrame(coord).duplicated()==True)[0].shape[0] !=0:
        raise NotImplementedError()
    
    return img1, img10,coord, label

def normalize_pixel_value(img):
    return img / 255.0

def main():

    opt = parse_option()
    
    total_img1, total_img10, total_coord, total_label = read_matfile_and_split(opt.mat_file_path)

    total_img1, total_img10, total_coord, total_label = check_negative_label(total_img1, total_img10, total_coord, total_label, dataset='Total')
    total_img1, total_img10, total_coord, total_label = check_duplicated(total_img1, total_img10, total_coord, total_label, opt,dataset='Total')

    #total_img10 = normalize_pixel_value(total_img10)
    

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



