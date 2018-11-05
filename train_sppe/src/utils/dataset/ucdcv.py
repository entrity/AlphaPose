# -----------------------------------------------------
# -----------------------------------------------------

# Usage:
# cd AlphaPose/train_sppe/src/
# mkdir ../exp
# python train.py --dataset ucd --loadModel ../../models/sppe/duc_se.pth
# python train.py --dataset ucd --loadModel ../../models/yolo/yolov3-spp.weights

import os
import numpy as np
from functools import reduce

import torch.utils.data as data
from ..pose import generateSampleBox
from opt import opt

import pdb

IMGSET = 'ucd'

class UcdCV(data.Dataset):
    def __init__(self, train=True, sigma=1,
                 kp_fmt='coco',
                 scale_factor=(0.2, 0.3), rot_factor=40, label_type='Gaussian'):
        self.img_folder = '../data/ucdcv/images'    # root image folders
        self.is_train = train           # training set or test set
        self.inputResH = opt.inputResH
        self.inputResW = opt.inputResW
        self.outputResH = opt.outputResH
        self.outputResW = opt.outputResW
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.label_type = label_type

        # Determine which set of keypoints to use
        if kp_fmt == 'coco':
            self.keypoint_indices = np.array([
                True, # "Nose"
                True, # "LEye"
                True, # "REye"
                True, # "LEar"
                True, # "REar"
                False, # "Head"
                False, # "Neck"
                True, # "LShoulder"
                True, # "RShoulder"
                True, # "LElbow"
                True, # "RElbow"
                True, # "LWrist"
                True, # "RWrist"
                False, # "Thrx"
                True, # "LHip"
                True, # "RHip"
                False, # "Pelv"
                True, # "LKnee"
                True, # "Rknee"
                True, # "LAnkle"
                True, # "RAnkle"
            ])
            assert self.keypoint_indices.sum() == 17, self.keypoint_indices.sum()
        elif kp_fmt == 'mpii':
            self.keypoint_indices = np.array([
                False, # "Nose"
                False, # "LEye"
                False, # "REye"
                False, # "LEar"
                False, # "REar"
                True, # "Head"
                True, # "Neck"
                True, # "LShoulder"
                True, # "RShoulder"
                True, # "LElbow"
                True, # "RElbow"
                True, # "LWrist"
                True, # "RWrist"
                True, # "Thrx"
                True, # "LHip"
                True, # "RHip"
                True, # "Pelv"
                True, # "LKnee"
                True, # "Rknee"
                True, # "LAnkle"
                True, # "RAnkle"
            ])
            assert self.keypoint_indices.sum() == 16, self.keypoint_indices.sum()
        else:
            raise Exception('Illegal keypoint format : %s' % str(kp_fmt))
        self.n_joints = self.keypoint_indices.sum()

        # Get annotations from annotations file
        regex_types = [('fpath', np.str_, 32)]
        regex       = r'(\S+)'
        regex_types += [ ('bb%s%d' % (v,i), np.float32) for i in range(1,3) for v in ('x','y') ]
        regex       += r'\t([\d.]+)' * 4
        regex_types += [ ('kp%s%d' % (v,i), np.float32) for i in range(21) for v in ('x','y')]
        regex       += r'\t([\d.]+)' * 2 * 21
        self.meta        = np.fromregex('../data/ucdcv/annotations.tsv', regex, regex_types)
        headers          = self.meta.dtype.names
        self.all_impath  = self.meta['fpath']
        self.all_bbs     = np.stack( [self.meta[key] for key in headers[1:5]] ).transpose(1,0)
        kpx              = np.stack( [self.meta[key] for key in headers[5::2]] )
        kpy              = np.stack( [self.meta[key] for key in headers[6::2]] )
        self.all_kps     = np.stack( [kpx, kpy] ).transpose(2,1,0)
        print(self.all_bbs.shape)
        print(self.all_kps.shape)
        print(self.keypoint_indices.shape)
        # import IPython
        # IPython.embed()
        # Partition into train vs test
        n = len(self.meta)
        train_test_boundary = int(n * .9)
        self.rows        = np.arange(train_test_boundary) if self.is_train else np.arange(train_test_boundary, n)

    def __getitem__(self, index):
        pindex   = self.rows[index]
        img_path = os.path.join( '../data/ucdcv', self.all_impath[pindex] )

        bndbox   = self.all_bbs[pindex, :]
        part     = self.all_kps[pindex, self.keypoint_indices, :]

        metaData = generateSampleBox(img_path, bndbox, part, self.n_joints,
                                     IMGSET, self.scale_factor, self, train=self.is_train)

        inp, out, setMask = metaData

        return inp, out, setMask, IMGSET

    def __len__(self):
        return len(self.rows)

# Usage:
#   python -m utils.dataset.ucdcv
if __name__ == '__main__':
    data = UcdCV(train=True)
    from .coco import Mscoco
    coco = Mscoco()
    import IPython
    IPython.embed()