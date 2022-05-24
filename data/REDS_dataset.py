import os
import random
from torch.utils.data import Dataset
from data.utils import read_img
import numpy as np
import cv2
import torch


class REDSDataset(Dataset):
   
    def __init__(self, opt):
        super(REDSDataset, self).__init__()
        self.path = opt['dataset']['path']
        self.N = opt['dataset']['N']
        self.GT_size = opt['dataset']['GT_size']
        self.LR_size = opt['dataset']['LR_size']
        self.scale = opt['dataset']['GT_size'] // opt['dataset']['LR_size']

        self.data = []
        v_list = os.listdir(self.path)
        for v in v_list:
            vpath = os.path.join(self.path, v)
            num_keys = len(os.listdir(vpath))
            for i in range(num_keys):
                self.data.append((v, i))


    def __getitem__(self, index):
        v, key = self.data[index]
        vpath = os.path.join(self.path, v)
        num_keys = len(os.listdir(vpath))
        
        img_GT = read_img(vpath, key, dataset_name='REDS') # HWC, BGR, normalized
        H,W = img_GT.shape[0], img_GT.shape[1]
        assert min(H,W) >= self.GT_size
        # extract patch (random crop)
        rnd_h = random.randint(0, H-self.GT_size)
        rnd_w = random.randint(0, W-self.GT_size)
        img_GT = img_GT[rnd_h:rnd_h+self.GT_size, rnd_w:rnd_w+self.GT_size, :]

        img_LRs = []
        for i in range(key-self.N, key+self.N+1):
            j = min(max(0, i), num_keys-1)
            img_HR = read_img(vpath, j, dataset_name='REDS')
            assert img_HR.shape[0]==H and img_HR.shape[1]==W
            # extract patch
            img_HR = img_HR[rnd_h:rnd_h+self.GT_size, rnd_w:rnd_w+self.GT_size, :]
            # HR to LR
            img_HR = cv2.GaussianBlur(img_HR, (5,5), 0)
            img_LR = img_HR[::self.scale, ::self.scale, :]
            img_LRs.append(img_LR)
        img_LRs = np.stack(img_LRs, axis=0) # NHWC

        img_GT = img_GT[:, :, [2,1,0]] # RGB
        img_LRs = img_LRs[:, :, :, [2,1,0]] # RGB
        img_GT = np.ascontiguousarray(np.transpose(img_GT, (2,0,1))) # CHW
        img_LRs = np.ascontiguousarray(np.transpose(img_LRs, (0,3,1,2))) # NCHW
        img_GT = torch.from_numpy(img_GT)
        img_LRs = torch.from_numpy(img_LRs)
        return {'LRs': img_LRs, 'GT': img_GT}


    def __len__(self):
        return len(self.data)


