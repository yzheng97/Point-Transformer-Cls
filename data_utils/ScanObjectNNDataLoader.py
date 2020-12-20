import numpy as np
import warnings
import os
import h5py
from tqdm import tqdm
import pickle
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')



def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def convert_to_binary_mask(masks):
    binary_masks = []
    for i in range(masks.shape[0]):
        binary_mask = np.ones(masks[i].shape)
        bg_idx = np.where(masks[i,:]==-1)
        binary_mask[bg_idx] = 0
        binary_masks.append(binary_mask)
    binary_masks = np.array(binary_masks)
    return binary_masks


class ScanObjectNNDataLoader(Dataset):
    def __init__(self, root,  npoint=1024, split='train', uniform=False, normal_channel=True, cache_size=15000):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        #self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        #self.cat = [line.rstrip() for line in open(self.catfile)]
        #self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel

        #shape_ids = {}
        #shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
        #shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        #shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple
        data = h5py.File(self.root + '/' + split + '_objectdataset_augmentedrot_scale75.h5', 'r')
        self.points = data['data'][:]
        self.labels = data['label'][:]
        self.masks = convert_to_binary_mask(data['mask'][:])
        print('The size of %s data is %d'%(split,self.points.shape[0]))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple
        

    def __len__(self):
        return self.points.shape[0]

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls, mask = self.cache[index]
        else:
            point_set = self.points[index]
            cls = self.labels[index]
            mask = self.masks[index]
            #fn = self.datapath[index]
            #cls = self.classes[self.datapath[index][0]]
            #cls = np.array([cls]).astype(np.int32)
            #point_set = self.all_data[fn[1]]
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints,:]
                mask = mask[0:self.npoints]

            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            if not self.normal_channel:
                point_set = point_set[:, 0:3]

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, mask)

        return point_set, cls, mask

    def __getitem__(self, index):
        return self._get_item(index)




if __name__ == '__main__':
    import torch

    data = ModelNetDataLoader('/data/modelnet40_normal_resampled/',split='train', uniform=False, normal_channel=True,)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point,label in DataLoader:
        print(point.shape)
        print(label.shape)
