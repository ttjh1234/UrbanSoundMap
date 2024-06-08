import numpy as np
import h5py

file_object = h5py.File('./assets/newdata'+'/test_region_{}.h5py'.format('DJ'), 'r')

# def srs_group(total_size, sample_size):
#     sample_idx = np.random.choice(range(total_size),size=sample_size, replace=False)
#     sample_idx = np.sort(sample_idx)
     
#     return sample_idx

# sample_idx = srs_group(file_object['test_label'].shape[0], 1000)
# sample_img10 = file_object['test_img10'][sample_idx]
# sample_img1 = file_object['test_img1'][sample_idx]
# sample_label = file_object['test_label'][sample_idx]

# np.save('./assets/sample/sample_img10',sample_img10)
# np.save('./assets/sample/sample_img1',sample_img1)
# np.save('./assets/sample/sample_label',sample_label)

total_img10 = file_object['test_img10']
total_img1 = file_object['test_img1']
total_label = file_object['test_label']
total_coord = file_object['test_coord']

idx = np.load('./assets/newdata/dj_train_idx.npy')
idx = np.sort(idx)

total_img10 = np.array(total_img10)[idx]
total_img1 = np.array(total_img1)[idx]
total_label = np.array(total_label)[idx]
total_coord= np.array(total_coord)[idx]

np.save('./assets/newdata_dj/dj_total_label',total_label)
np.save('./assets/newdata_dj/dj_total_coord',total_coord)
np.save('./assets/newdata_dj/dj_total_img10',total_img10)
np.save('./assets/newdata_dj/dj_total_img1',total_img1)


