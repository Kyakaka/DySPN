"""
    Non-Local Spatial Propagation Network for Depth Completion
    Jinsun Park, Kyungdon Joo, Zhe Hu, Chi-Kuei Liu and In So Kweon

    European Conference on Computer Vision (ECCV), Aug 2020

    Project Page : https://github.com/zzangjinsun/NLSPN_ECCV20
    Author : Jinsun Park (zzangjinsun@kaist.ac.kr)

    ======================================================================

    NYU Depth V2 Dataset Helper
"""


import os
import warnings
import numpy as np
import json
import h5py

from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import settings_nyu_s as settings
warnings.filterwarnings("ignore", category=UserWarning)


class NYU():
    def __init__(self, mode):
        super(NYU, self).__init__()

        self.mode = mode

        assert mode in ['train','val','test'], "NotImplementedError"

        # For NYUDepthV2, crop size is fixed
        height, width = (240, 320)
        crop_size = (228, 304)

        self.height = height
        self.width = width
        self.crop_size = crop_size

        # Camera intrinsics [fx, fy, cx, cy]
        self.K = torch.Tensor([
            5.1885790117450188e+02 / 2.0,
            5.1946961112127485e+02 / 2.0,
            3.2558244941119034e+02 / 2.0 - 8.0,
            2.5373616633400465e+02 / 2.0 - 6.0
        ])

        self.augment = settings.augment

        if mode == "train":
            self.sample_list = self.read_paths(settings.split_train)
            self.path = "train/{}.h5"
        elif mode == "val":
            self.sample_list = self.read_paths(settings.split_val)
            self.path = "val/{}.h5"
        else:
            self.sample_list = self.read_paths(settings.split_val)
            self.path = "val/{}.h5"

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        path_file = os.path.join(settings.NYU_PATH, self.path.format(self.sample_list[idx]))

        f = h5py.File(path_file, 'r')
        rgb_h5 = f['rgb'][:].transpose(1, 2, 0)
        dep_h5 = f['depth'][:]

        rgb = Image.fromarray(rgb_h5, mode='RGB')
        dep = Image.fromarray(dep_h5.astype('float32'), mode='F')

        if self.augment and self.mode == 'train':
            _scale = np.random.uniform(1.0, 1.5)
            scale = int(self.height * _scale)
            degree = np.random.uniform(-5.0, 5.0)
            flip = np.random.uniform(0.0, 1.0)

            if flip > 0.5:
                rgb = TF.hflip(rgb)
                dep = TF.hflip(dep)

            # rgb = TF.rotate(rgb, angle=degree, resample=Image.NEAREST)
            # dep = TF.rotate(dep, angle=degree, resample=Image.NEAREST)
            rgb = TF.rotate(rgb, angle=degree)
            dep = TF.rotate(dep, angle=degree)
            t_rgb = T.Compose([
                T.Resize(scale),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                T.CenterCrop(self.crop_size),
                T.ToTensor(),
                # T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

            t_dep = T.Compose([
                T.Resize(scale),
                T.CenterCrop(self.crop_size),
                self.ToNumpy(),
                T.ToTensor()
            ])

            rgb = t_rgb(rgb)
            dep = t_dep(dep)

            dep = dep / _scale

            K = self.K.clone()
            K[0] = K[0] * _scale
            K[1] = K[1] * _scale
        else:
            t_rgb = T.Compose([
                T.Resize(self.height),
                T.CenterCrop(self.crop_size),
                T.ToTensor(),
                # T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            t_dep = T.Compose([
                T.Resize(self.height),
                T.CenterCrop(self.crop_size),
                self.ToNumpy(),
                T.ToTensor()
            ])

            rgb = t_rgb(rgb)
            dep = t_dep(dep)

            K = self.K.clone()

        dep_sp = self.get_sparse_depth(dep, settings.num_sample)

        output = {'rgb': rgb, 'dep': dep_sp, 'gt': dep, 'K': K}

        return output
    def read_paths(self, filepath):
        '''
        Reads a newline delimited file containing paths

        Arg(s):
            filepath : str
                path to file to be read
        Return:
            list[str] : list of paths
        '''

        path_list = []
        with open(filepath) as f:
            while True:
                path = f.readline().rstrip('\n')
                # If there was nothing to read
                if path == '':
                    break
                path_list.append(path)

        return path_list

    def get_sparse_depth(self, dep, num_sample):
        channel, height, width = dep.shape

        assert channel == 1

        idx_nnz = torch.nonzero(dep.view(-1) > 0.0001, as_tuple=False)

        num_idx = len(idx_nnz)
        idx_sample = torch.randperm(num_idx)[:num_sample]

        idx_nnz = idx_nnz[idx_sample[:]]

        mask = torch.zeros((channel*height*width))
        mask[idx_nnz] = 1.0
        mask = mask.view((channel, height, width))

        dep_sp = dep * mask.type_as(dep)

        return dep_sp
    # A workaround for a pytorch bug
    # https://github.com/pytorch/vision/issues/2194

    class ToNumpy:
        def __call__(self, sample):
            return np.array(sample)

if __name__ == '__main__':
    print(" ")