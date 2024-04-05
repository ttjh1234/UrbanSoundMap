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
    parser.add_argument('--mat_file_path', type=str, default = "./assets/GJ_whole_city.mat",help='Mat File Path')
    parser.add_argument('--seed', type=int, default = 0,help='File Seed' )
    parser.add_argument('--save_flag', type=int, default = 0,help='File Save Flag 0 is no save, if you want to save, use 1' )
    opt = parser.parse_args()
    
    return opt


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

def check_over_label(img1,img10,label,coord):
    over_index = np.where(label==1000)[0]

    img1 = np.delete(img1, over_index, axis=0)
    img10 = np.delete(img10, over_index, axis=0)
    label = np.delete(label, over_index, axis=0)
    coord = np.delete(coord, over_index, axis=0)

    return img1, img10, label, coord


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def main():

    opt = parse_option()
    
    total_img1, total_img10, total_coord, total_label = read_matfile_and_split(opt.mat_file_path)

    total_img1, total_img10, total_coord, total_label = check_negative_label(total_img1, total_img10, total_coord, total_label, dataset='Total')
    total_img1, total_img10, total_coord, total_label = check_duplicated(total_img1, total_img10, total_coord, total_label, opt,dataset='Total')

    #total_img10 = normalize_pixel_value(total_img10)
        
    set_random_seed(opt.seed)
    tr_idx, va_idx = choose_region(opt.seed)
    print(opt.seed, "- Train ", tr_idx)
    print(opt.seed, "- Test ", va_idx)
    
    train_img1, train_img10, train_label, train_coord = parsing_index(total_img10, total_label, total_coord, tr_idx, total_img1)
    test_img1, test_img10, test_label, test_coord = parsing_index(total_img10, total_label, total_coord, va_idx, total_img1)
    del total_img10
    del total_label
    del total_coord
    del total_img1
    
    
    # Get seed mean, std
    # Whole image
    train_img10_mean = np.mean(np.mean(train_img10,axis=(1,2)))/255.0
    stdlist = np.zeros((0,))
    for i in range(10):
        if i==9:
            sample = train_img10[(train_img10.shape[0]//10*9):,:,:]
            smean = np.mean(sample,axis=(1,2),keepdims=True)
            stdlist = np.concatenate([stdlist,(np.sqrt(np.mean((sample -smean)**2,axis=(1,2))))],axis=0)
        else:
            sample = train_img10[(train_img10.shape[0]//10 * i):(train_img10.shape[0]//10 * (i+1)),:,:]
            smean = np.mean(sample,axis=(1,2),keepdims=True)
            stdlist = np.concatenate([stdlist,(np.sqrt(np.mean((sample -smean)**2,axis=(1,2))))],axis=0)

    train_img10_std = np.mean(stdlist)/255.0
    
    train_img1_mean = np.mean(np.mean(train_img1,axis=(1,2)))/255.0
    stdlist = np.zeros((0,))
    for i in range(10):
        if i==9:
            sample = train_img1[(train_img1.shape[0]//10*9):,:,:]
            smean = np.mean(sample,axis=(1,2),keepdims=True)
            stdlist = np.concatenate([stdlist,(np.sqrt(np.mean((sample -smean)**2,axis=(1,2))))],axis=0)
        else:
            sample = train_img1[(train_img1.shape[0]//10 * i):(train_img1.shape[0]//10 * (i+1)),:,:]
            smean = np.mean(sample,axis=(1,2),keepdims=True)
            stdlist = np.concatenate([stdlist,(np.sqrt(np.mean((sample -smean)**2,axis=(1,2))))],axis=0)

    train_img1_std = np.mean(stdlist)/255.0

    build = np.where(train_img10<90,train_img10, 255)
    build_mean = np.mean(np.mean(build,axis=(1,2)))/255.0
    stdlist = np.zeros((0,))
    for i in range(10):
        if i==9:
            sample = build[(build.shape[0]//10*9):,:,:]
            smean = np.mean(sample,axis=(1,2),keepdims=True)
            stdlist = np.concatenate([stdlist,(np.sqrt(np.mean((sample -smean)**2,axis=(1,2))))],axis=0)
        else:
            sample = build[(build.shape[0]//10 * i):(build.shape[0]//10 * (i+1)),:,:]
            smean = np.mean(sample,axis=(1,2),keepdims=True)
            stdlist = np.concatenate([stdlist,(np.sqrt(np.mean((sample -smean)**2,axis=(1,2))))],axis=0)
    del build

    build_std = np.mean(stdlist)/255.0
    
    road = np.where(train_img10>=90,train_img10, 255)
    road_mean = np.mean(np.mean(road,axis=(1,2)))/255.0
    stdlist = np.zeros((0,))
    for i in range(10):
        if i==9:
            sample = road[(road.shape[0]//10*9):,:,:]
            smean = np.mean(sample,axis=(1,2),keepdims=True)
            stdlist = np.concatenate([stdlist,(np.sqrt(np.mean((sample -smean)**2,axis=(1,2))))],axis=0)
        else:
            sample = road[(road.shape[0]//10 * i):(road.shape[0]//10 * (i+1)),:,:]
            smean = np.mean(sample,axis=(1,2),keepdims=True)
            stdlist = np.concatenate([stdlist,(np.sqrt(np.mean((sample -smean)**2,axis=(1,2))))],axis=0)
    del road
    
    road_std = np.mean(stdlist)/255.0
    
    mean_vec = [train_img1_mean, build_mean, road_mean, train_img10_mean]
    std_vec = [train_img1_std, build_std, road_std, train_img10_std]
    summary_statistics = pd.DataFrame([mean_vec, std_vec],columns = ['expand','build','road','origin'])
    
    print("Mean Vec : ", mean_vec)
    print("Std Vec : ", std_vec)
    
    hdf5_file = './assets/total_data_{}.h5py'.format(opt.seed)
    if opt.save_flag==1:
        with h5py.File(hdf5_file, 'w') as hf:
            hf.create_dataset('train_img10',data=train_img10,chunks=(10000, 100, 100))
            hf.create_dataset('train_img1',data=train_img1, chunks=(10000,100,100))
            hf.create_dataset('train_label',data=train_label, chunks=(20000,))
            hf.create_dataset('train_coord',data=train_coord, chunks=(20000,2))
            hf.create_dataset('test_img10',data=test_img10,chunks=(10000, 100, 100))
            hf.create_dataset('test_img1',data=test_img1, chunks=(10000,100,100))
            hf.create_dataset('test_label',data=test_label, chunks=(20000,))
            hf.create_dataset('test_coord',data=test_coord, chunks=(20000,2))

        summary_statistics.to_csv('./assets/total_data_summary_{}.csv'.format(opt.seed),index=False)
    
    

if __name__ == '__main__':
    main()



