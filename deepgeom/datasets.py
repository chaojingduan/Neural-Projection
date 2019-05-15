'''
PointCloudDataset in deepgeom

author  : cfeng
created : 1/26/18 11:21 PM
'''

import os
import sys
import argparse
import glog as logger

from torch.utils.data import Dataset
import numpy as np

class PointCloudOldDataset(Dataset):

    def __init__(self, pkl_path, shuffle_point_order='no'):
        self.shuffle_point_order = shuffle_point_order

        logger.info('loading: '+pkl_path)
        # with open(pkl_path) as f:

        raw_data = np.load(pkl_path, encoding='bytes').item()
        self.all_data = raw_data[b'data'] #[BxNx3]
        # self.all_isomap = raw_data[b'isomap']
        if shuffle_point_order=='preprocess':
            for i in xrange(self.all_data.shape[0]):
                np.random.shuffle(self.all_data[i])
        self.all_label = np.asarray(raw_data[b'label'], dtype=np.int64)

        logger.info('pkl loaded: data '+str(self.all_data.shape)+', label '+str(self.all_label.shape))
        logger.check_eq(len(self.all_data.shape), 3,
                        'data field should of size BxNx3!')
        logger.check_eq(self.all_data.shape[-1], 3,
                        'data field the last dimension size should be 3!')
        logger.check_eq(len(self.all_label.shape), 1,
                        'label field should be one dimensional!')
        logger.check_eq(self.all_data.shape[0], self.all_label.shape[0],
                        'data field and label field should have the same size along the first dimension!')


    def __len__(self):
        return self.all_data.shape[0]

    def __getitem__(self, idx):
        if self.shuffle_point_order=='online':
            np.random.shuffle(self.all_data[idx])
        return {'data':self.all_data[idx], 'label':self.all_label[idx]}


class PointCloudDataset(Dataset):

    def __init__(self, pkl_path, shuffle_point_order='no'):
        self.shuffle_point_order = shuffle_point_order

        logger.info('loading: '+pkl_path)
        # with open(pkl_path) as f:

        raw_data = np.load(pkl_path, encoding='bytes')#.item()
        #self.all_data = raw_data[b'data'] #[BxNx3]
        #print('raw data shape: ',raw_data.shape)
        self.all_data = raw_data
        # self.all_isomap = raw_data[b'isomap']
        if shuffle_point_order=='preprocess':
            for i in xrange(self.all_data.shape[0]):
                np.random.shuffle(self.all_data[i])
        #self.all_label = np.asarray(raw_data[b'label'], dtype=np.int64)

        #logger.info('pkl loaded: data '+str(self.all_data.shape)+', label '+str(self.all_label.shape))
        #print(self.all_data.shape)
        logger.check_eq(len(self.all_data.shape), 3,
                        'data field should of size BxNx3!')
        logger.check_eq(self.all_data.shape[-1], 3,
                        'data field the last dimension size should be 3!')
        #logger.check_eq(len(self.all_label.shape), 1,
        #                'label field should be one dimensional!')
        #logger.check_eq(self.all_data.shape[0], self.all_label.shape[0],
        #                'data field and label field should have the same size along the first dimension!')


    def __len__(self):
        return self.all_data.shape[0]


    def __getitem__(self, idx):
        if self.shuffle_point_order=='online':
            np.random.shuffle(self.all_data[idx])
        return {'data':self.all_data[idx]}#, 'label':self.all_label[idx]}


class PointCloudDenoiseDataset(Dataset):

    def __init__(self, pkl_path, shuffle_point_order='no'):
        self.shuffle_point_order = shuffle_point_order

        logger.info('loading: '+pkl_path)
        # with open(pkl_path) as f:

        raw_data = np.load(pkl_path, encoding='bytes').item()
        self.all_data = raw_data[b'data'] #[B x N x 3]
        self.all_isomap = raw_data[b'pure'] # [B x N x 2]
        print("pure point cloud size", self.all_isomap.shape)
        if shuffle_point_order=='preprocess':
            for i in xrange(self.all_data.shape[0]):
                np.random.shuffle(self.all_data[i])
        self.all_label = np.asarray(raw_data[b'label'], dtype=np.int64)

        logger.info('pkl loaded: data '+str(self.all_data.shape)+', label '+str(self.all_label.shape))

        logger.check_eq(len(self.all_data.shape), 3,
                        'data field should of size BxNx3!')
        logger.check_eq(len(self.all_isomap.shape), 3,
                        'isomap field should of size BxNx2!')
        logger.check_eq(self.all_data.shape[-1], 3,
                        'data field the last dimension size should be 3!')
        logger.check_eq(len(self.all_label.shape), 1,
                        'label field should be one dimensional!')
        logger.check_eq(self.all_data.shape[0], self.all_label.shape[0],
                        'data field and label field should have the same size along the first dimension!')
        logger.check_eq(self.all_data.shape[0], self.all_isomap.shape[0],
                        'data field and isomap field should have the same size along the first dimension!')


    def __len__(self):
        return self.all_data.shape[0]


    def __getitem__(self, idx):
        if self.shuffle_point_order=='online':
            np.random.shuffle(self.all_data[idx])
        return {'data':self.all_data[idx], 'isomap':self.all_isomap[idx] , 'label':self.all_label[idx]}




def main(args):
    import utils

    modelnet = PointCloudDataset(pkl_path='../../data/modelNet_test.pkl',shuffle_point_order='online')
    for i, ith in enumerate(modelnet):
        print('{:d}: {:d}'.format(i, ith['label']))

        X = ith['data']
        rdir = np.random.rand(3,)
        rdir/= np.linalg.norm(rdir.reshape(-1))
        clr = (X.dot(rdir)).reshape(-1)
        clr = (clr-clr.min())/(clr.max()-clr.min())
        clr = np.round(clr*20)
        # clr = xrange(X.shape[0])
        fig = utils.vis_pts(X, clr=clr, cmap='tab20')
        utils.plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0])

    args = parser.parse_args(sys.argv[1:])
    args.script_folder = os.path.dirname(os.path.abspath(__file__))

    main(args)
