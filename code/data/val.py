import os

from data import common
from data import srdata

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

class VAL(srdata.SRData):
    def __init__(self, args, train=True):
        super(VAL, self).__init__(args, train, benchmark=True)

    def _scan(self):
        list_hr = []
        list_lr = [[] for _ in self.scale]
        for entry in os.scandir(self.dir_hr):
            filename = os.path.splitext(entry.name)[0]
            list_hr.append(os.path.join(self.dir_hr, filename + self.ext))
            for si, s in enumerate(self.scale):
                list_lr[si].append(os.path.join(
                    self.dir_lr,
                    '{}{}'.format(filename, self.ext)
                ))

        list_hr.sort()
        for l in list_lr:
            l.sort()

        return list_hr, list_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'val')
        self.dir_hr = os.path.join(self.apath, 'label_png')
        self.dir_lr = os.path.join(self.apath, 'input_png')
        self.ext = '.png'
