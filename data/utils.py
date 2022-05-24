import os
import numpy as np
import cv2


def read_img(vpath, i, dataset_name='REDS'):
    if dataset_name == 'REDS':
        ipath = os.path.join(vpath, '{:08d}'.format(i)+'.png')
    else:
        raise NotImplementedError

    img = cv2.imread(ipath) # HWC, BGR
    img = img.astype(np.float32) / 255.
    return img
