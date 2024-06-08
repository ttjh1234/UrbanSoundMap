import h5py
import torch
import numpy as np
from tqdm import tqdm
import argparse
from tqdm import tqdm
import torch.nn as nn
import h5py

# base to coord 
# we parse 9 region corresponding to 1 img.
# Consider Non overlap region
# Finally, we return 1 img per 9label, center coord, 
# Non overlap -> total 250

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--filepath', type=str, default='./assets/newdata/', help='File path')
    parser.add_argument('--neighbor', type=int, default=3, choices=[3,5,7], help='Neighbor')
    opt = parser.parse_args()
    
    return opt
def get_neighbor_noise(idx,noise,neighbor):
    target = noise[(idx[0]-neighbor//2):(idx[0]+neighbor//2+1), (idx[1]-neighbor//2):(idx[1]+neighbor//2+1)]
    return target.T.reshape(1,-1)


def main():
    opt = parse_option()
    file_object = h5py.File('./assets/total_data/'+'/total_data_0.h5py', 'r')
    neighbor = opt.neighbor

    use_train_coord = np.load('./assets/newdata/train_coord.npy')
    use_test_coord = np.load('./assets/newdata/test_coord.npy')
    use_coord = np.concatenate([use_train_coord, use_test_coord],axis=0)
    del use_train_coord
    del use_test_coord

    test_coord = np.array(file_object['test_coord'])
    train_coord = np.array(file_object['train_coord'])
    test_label = np.array(file_object['test_label'])
    train_label = np.array(file_object['train_label'])

    coord = np.concatenate([train_coord,test_coord],axis=0)
    label = np.concatenate([train_label, test_label],axis=0)
    del test_coord
    del train_coord
    del test_label
    del train_label

    xmin = np.min(coord[:,0])
    xmax = np.max(coord[:,0])
    ymin = np.min(coord[:,1])
    ymax = np.max(coord[:,1])

    oriented_coord = coord - np.array([xmin,ymin])

    board = np.zeros((int(xmax-xmin+1),int(ymax-ymin+1)))
    noise = np.zeros((int(xmax-xmin+1),int(ymax-ymin+1)))

    board[oriented_coord[:,0].astype(int),oriented_coord[:,1].astype(int)] = 1
    noise[oriented_coord[:,0].astype(int),oriented_coord[:,1].astype(int)] = label 

    board_ts= torch.tensor(board,dtype=torch.float)

    myconv = nn.Conv2d(1,1,neighbor,1,0,bias=False)
    nn.init.constant_(myconv.weight, 1)
    myconv.weight.requires_grad = False

    result = myconv(board_ts.unsqueeze(0))
    result = result.squeeze(0)

    print(torch.where(result==(neighbor**2))[0].shape)
    # 197682 Sample 
    row,col = torch.where(result == (neighbor**2))
    idx = torch.concat([row.unsqueeze(1),col.unsqueeze(1)],dim=1)

    origin_idx = idx  + torch.tensor([[(neighbor-1)//2,(neighbor-1)//2]])
    #origin_idx = idx * neighbor + torch.tensor([[1, 1]])
    
    use_idx = use_coord - np.array([xmin,ymin])

    temp_use_idx = use_idx[:,0]*10000 + use_idx[:,1]
    temp_origin_idx = origin_idx[:,0]*10000 + origin_idx[:,1]
    
    isin_idx = np.in1d(temp_origin_idx, temp_use_idx)
    origin_idx = origin_idx[isin_idx,:]

    print("N data : ", origin_idx.shape)
    # 1,1 <- 0,0
    # 1,2 <- 0,1
    # Origin coordinate = Oriented coordinate + min coordinate

    # origin idx represents center coordinates. 
    # Based on oriented coordinate idx, extract 9 noise point centered by origin idx containing center noise.
    # And then transform oriented to origin, after add min coord, find corresponding images.
    # That image is considered as input data.


    target_list = []
    for i in range(origin_idx.shape[0]):
        target_list.append(get_neighbor_noise(origin_idx[i],noise,neighbor))

    n_row = len(target_list)
    target_arr = np.array(target_list).reshape(n_row,neighbor**2)
    target_coord = origin_idx.numpy() + np.array([[xmin,ymin]])

    # Label , Coord save
    np.save('./assets/multiple/multiple_overlap_noise_{}'.format(neighbor),target_arr)
    np.save('./assets/multiple/multiple_overlap_coord_{}'.format(neighbor),target_coord)

    target_coord = target_coord.astype('int')


    mimg1 = np.zeros((n_row,100,100))
    mimg10 = np.zeros((n_row,100,100))
    for i in tqdm(range(target_coord.shape[0])):
        img_idx = np.where((coord[:,0]== target_coord[i,0])&(coord[:,1]== target_coord[i,1]))[0]
        if img_idx >=1631172:
            simg = file_object['test_img10'][img_idx-1631172].reshape(1,100,100)
            simg1 = file_object['test_img1'][img_idx-1631172].reshape(1,100,100)
        else:
            simg = file_object['train_img10'][img_idx].reshape(1,100,100)
            simg1 = file_object['train_img1'][img_idx].reshape(1,100,100)

        mimg1[i] =simg1
        mimg10[i] = simg


    # Image Save
    np.save('./assets/multiple/multiple_overlap_img1_{}'.format(neighbor),mimg1)
    np.save('./assets/multiple/multiple_overlap_img10_{}'.format(neighbor),mimg10)


    # Visualization 
    # import matplotlib.pyplot as plt
    # import matplotlib.patches as patches

    # plt.imshow(mimg10[90144],cmap='gray')
    # plt.imshow(mimg1[90144],cmap='gray',vmin=0,vmax=255)
    # shp=patches.Circle((39.5,39.5), radius=0.5, color='b')
    # plt.gca().add_patch(shp)
    # shp=patches.Circle((49.5,39.5), radius=0.5, color='r')
    # plt.gca().add_patch(shp)
    # shp=patches.Circle((59.5,59.5), radius=0.5, color='b')
    # plt.gca().add_patch(shp)
    # shp=patches.Circle((59.5,49.5), radius=0.5, color='r')
    # plt.gca().add_patch(shp)
    # shp=patches.Circle((59.5,39.5), radius=0.5, color='b')
    # plt.gca().add_patch(shp)
    # shp=patches.Circle((49.5,59.5), radius=0.5, color='r')
    # plt.gca().add_patch(shp)
    # shp=patches.Circle((39.5,59.5), radius=0.5, color='b')
    # plt.gca().add_patch(shp)
    # shp=patches.Circle((39.5,49.5), radius=0.5, color='r')
    # plt.gca().add_patch(shp)
    # shp=patches.Circle((49.5,49.5), radius=0.5, color='b')
    # plt.gca().add_patch(shp)
    # plt.show()
    # target_arr[90144]


if __name__=='__main__':
    main()