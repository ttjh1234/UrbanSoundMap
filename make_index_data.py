import numpy as np

def main():
    
    file_path = './assets/newdata/'
    timg1= np.load(file_path+'train_img1.npy')
    vimg1= np.load(file_path+'test_img1.npy')

    img = np.concatenate([timg1, vimg1],axis=0)

    del timg1
    del vimg1
        
    build = np.where(img<=90/255.0, img, 1.0)
    road = np.where(img>=90/255.0, img, 1.0)
    
    del img
    
    road_flag = np.where(np.sum(1-road,axis=(1,2))!=0, 1, 0)
    build_flag = np.where(np.sum(1-build,axis=(1,2))!=0, 1, 0)

    mask = (road_flag==1) & (build_flag==1)
    bothin = np.where(mask)[0] # Center idx
    
    mask = (road_flag==0) | (build_flag==0)
    bothnotin = np.where(mask)[0] # NonCenter idx
    
    np.save(file_path+'center_idx', bothin)
    np.save(file_path+'noncenter_idx', bothnotin)
    

if __name__ == '__main__':
    main()
