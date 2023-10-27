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
# from . import BaseDataset

from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
# from dataset import data_utils
import settings_void_s as settings
warnings.filterwarnings("ignore", category=UserWarning)
dataset_path = settings.VOID_PATH

class VOID():
    def __init__(self, mode):
        super(VOID, self).__init__()

        self.mode = mode
        self.dataset = settings.data_mode
        if mode != 'train' and mode != 'val' and mode != 'test':
            raise NotImplementedError

        height, width = (480, 640)
        self.crop_size = (480, 640)
        self.height = height
        self.width = width

        # Camera intrinsics [fx, fy, cx, cy]
        # self.K = torch.Tensor([
        #     5.1885790117450188e+02 / 2.0,
        #     5.1946961112127485e+02 / 2.0,
        #     3.2558244941119034e+02 / 2.0 - 8.0,
        #     2.5373616633400465e+02 / 2.0 - 6.0
        # ])

        # with open(self.args.split_json) as json_file:
        #     json_data = json.load(json_file)
        #     self.sample_list = json_data[mode]
        if self.mode == 'train':
            self.rgb_path_list = read_paths(
                dataset_path + self.dataset + '/train_image.txt')
            self.gt_path_list = read_paths(
                dataset_path + self.dataset + '/train_ground_truth.txt')
            self.depth_path_list = read_paths(
                dataset_path + self.dataset + '/train_sparse_depth.txt')
            self.intrinsics_paths = read_paths(
                dataset_path + self.dataset + '/train_intrinsics.txt')
        else:
            self.rgb_path_list = read_paths(
                dataset_path + self.dataset + '/test_image.txt')
            self.gt_path_list = read_paths(
                dataset_path + self.dataset + '/test_ground_truth.txt')
            self.depth_path_list = read_paths(
                dataset_path + self.dataset + '/test_sparse_depth.txt')
            self.intrinsics_paths = read_paths(
                dataset_path + self.dataset + '/test_intrinsics.txt')

    def __len__(self):
        return len(self.rgb_path_list)

    def __getitem__(self, idx):

        rgb = Image.open(os.path.join(dataset_path + self.rgb_path_list[idx]))
        gt = load_depth(os.path.join(self.gt_path_list[idx]))
        dep = load_depth(os.path.join( self.depth_path_list[idx]))
        gt = Image.fromarray(gt.astype('float32'), mode='F')
        dep = Image.fromarray(dep.astype('float32'), mode='F')
        intrinsics = np.loadtxt(os.path.join(dataset_path+ self.intrinsics_paths[idx])).astype(np.float32)
        if self.mode == 'train':
            # _scale = np.random.uniform(1.0, 1.5)
            # scale = np.int(self.height * _scale)
            # degree = np.random.uniform(-5.0, 5.0)
            # flip = np.random.uniform(0.0, 1.0)

            # if flip > 0.5:
            #     rgb = TF.hflip(rgb)
            #     dep = TF.hflip(dep)
            #     gt = TF.hflip(gt)
            #
            # rgb = TF.rotate(rgb, angle=degree, resample=Image.NEAREST)
            # dep = TF.rotate(dep, angle=degree, resample=Image.NEAREST)
            # gt = TF.rotate(gt, angle=degree, resample=Image.NEAREST)

            t_rgb = T.Compose([
                # T.Resize(scale),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                # T.CenterCrop(self.crop_size),
                T.ToTensor(),
                # T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

            t_dep = T.Compose([
                # T.Resize(scale),
                # T.CenterCrop(self.crop_size),
                # self.ToNumpy(),
                T.ToTensor()
            ])

            t_gt = T.Compose([
                # T.Resize(self.height),
                # self.ToNumpy(),
                T.ToTensor()
            ])

            rgb = t_rgb(rgb)
            dep = t_dep(dep)
            gt = t_gt(gt)

            # dep = dep / _scale
            # gt = gt / _scale
            # K = self.K.clone()
            # K[0] = K[0] * _scale
            # K[1] = K[1] * _scale
        else:
            t_rgb = T.Compose([
                # T.Resize(self.height),
                T.ToTensor(),
                # T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            t_dep = T.Compose([
                # T.Resize(self.height),
                # self.ToNumpy(),
                T.ToTensor()
            ])

            t_gt = T.Compose([
                # T.Resize(self.height),
                # self.ToNumpy(),
                T.ToTensor()
            ])

            rgb = t_rgb(rgb)
            dep = t_dep(dep)
            gt = t_gt(gt)
            # K = self.K.clone()

        # dep = self.get_sparse_depth2(gt, dep)
        # dep = self.get_sparse_depth(gt, self.args.num_sample)
        # output = {'rgb': rgb, 'dep': dep_sp, 'gt': dep, 'K': K}
        output = {'rgb': rgb, 'dep': dep, 'gt': gt, 'intrinsics': intrinsics}
        return output

    def get_sparse_depth(self, dep, num_sample):
        channel, height, width = dep.shape

        assert channel == 1

        idx_nnz = torch.nonzero(dep.view(-1) > 0.0001, as_tuple=False)

        num_idx = len(idx_nnz)
        idx_sample = torch.randperm(num_idx)[:num_sample]

        idx_nnz = idx_nnz[idx_sample[:]]

        mask = torch.zeros((channel * height * width))
        mask[idx_nnz] = 1.0
        mask = mask.view((channel, height, width))

        dep_sp = dep * mask.type_as(dep)

        return dep_sp

    def get_sparse_depth2(self, dep, dep2):

        dep_sp = dep * (dep2 > 0).type_as(dep)

        return dep_sp

def read_paths(filepath):
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

def load_depth(path, data_format='HW'):
    '''
    Loads a depth map from a 16-bit PNG file

    Arg(s):
        path : str
            path to 16-bit PNG file
        data_format : str
            HW, CHW, HWC
    Returns:
        numpy[float32] : depth map
    '''

    # Loads depth map from 16-bit PNG file
    z = np.array(Image.open(dataset_path + path), dtype=np.float32)

    # Assert 16-bit (not 8-bit) depth map
    z = z / 256.0
    z[z <= 0] = 0.0

    if data_format == 'HW':
        pass
    elif data_format == 'CHW':
        z = np.expand_dims(z, axis=0)
    elif data_format == 'HWC':
        z = np.expand_dims(z, axis=-1)
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))

    return z
