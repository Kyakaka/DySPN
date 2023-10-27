"""
    Non-Local Spatial Propagation Network for Depth Completion
    Jinsun Park, Kyungdon Joo, Zhe Hu, Chi-Kuei Liu and In So Kweon

    European Conference on Computer Vision (ECCV), Aug 2020

    Project Page : https://github.com/zzangjinsun/NLSPN_ECCV20
    Author : Jinsun Park (zzangjinsun@kaist.ac.kr)

    ======================================================================

    KITTI Depth Completion Dataset Helper
"""

from __future__ import print_function, division
import os
import numpy as np
import json
import random

from PIL import Image
import torch
import torchvision.transforms.functional as TF
import settings_kitti_s as settings


def read_depth(file_name):
    # loads depth map D from 16 bits png file as a numpy array,
    # refer to readme file in KITTI dataset
    assert os.path.exists(file_name), "file not found: {}".format(file_name)
    image_depth = np.array(Image.open(file_name))

    # Consider empty depth
    assert (np.max(image_depth) == 0) or (np.max(image_depth) > 255), \
        "np.max(depth_png)={}, path={}".format(np.max(image_depth), file_name)

    image_depth = image_depth.astype(np.float32) / 256.0
    return image_depth


# Reference : https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


class KITTIDC():
    def __init__(self, mode):
        super(KITTIDC, self).__init__()
        self.mode = mode
        assert mode in ['train', 'val', 'test'], "NotImplementedError"

        self.height = settings.patch_height
        self.width = settings.patch_width

        self.augment = settings.augment
        if mode == "train":
            self.sample_list = self.read_paths(settings.split_train)
            self.rgb_path = "image_raw/{}/{}/data/{}"
            self.depth_path = "data_depth_velodyne/train/{}/proj_depth/velodyne_raw/{}/{}"
            self.gt_path = "data_depth_annotated/train/{}/proj_depth/groundtruth/{}/{}"
            self.K_path = "data_calib/{}_calib/{}/calib_cam_to_cam.txt"
        elif mode == "val":
            self.sample_list = self.read_paths(settings.split_val)
            self.rgb_path = "data_depth_selection/depth_selection/val_selection_cropped/image/{}_image_{}.png"
            self.depth_path = "data_depth_selection/depth_selection/val_selection_cropped/velodyne_raw/{}_velodyne_raw_{}.png"
            self.gt_path = "data_depth_selection/depth_selection/val_selection_cropped/groundtruth_depth/{}_groundtruth_depth_{}.png"
            self.K_path = "data_depth_selection/depth_selection/val_selection_cropped/intrinsics/{}_image_{}.txt"
        else:
            self.sample_list = range(1000)
            self.rgb_path = "data_depth_selection/depth_selection/test_depth_completion_anonymous/image/{:010d}.png"
            self.depth_path = "data_depth_selection/depth_selection/test_depth_completion_anonymous/velodyne_raw/{:010d}.png"
            self.gt_path = "data_depth_selection/depth_selection/test_depth_completion_anonymous/velodyne_raw/{:010d}.png"
            self.K_path = "data_depth_selection/depth_selection/test_depth_completion_anonymous/intrinsics/{:010d}.txt"


    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        rgb, depth, gt, K = self._load_data(idx)

        if self.augment and self.mode == 'train':

            # Top crop if needed

            if settings.top_crop > 0:
                width, height = rgb.size
                rgb = TF.crop(rgb, settings.top_crop, 0,
                              height - settings.top_crop, width)
                depth = TF.crop(depth, settings.top_crop, 0,
                                height - settings.top_crop, width)
                gt = TF.crop(gt, settings.top_crop, 0,
                             height - settings.top_crop, width)
                K[3] = K[3] - settings.top_crop

            width, height = rgb.size

            _scale = np.random.uniform(1.0, 1.5)
            scale = int(height * _scale)
            degree = np.random.uniform(-5.0, 5.0)
            flip = np.random.uniform(0.0, 1.0)

            # Horizontal flip
            if flip > 0.5:
                rgb = TF.hflip(rgb)
                depth = TF.hflip(depth)
                gt = TF.hflip(gt)

                K[2] = width - K[2]

            # Rotation
            rgb = TF.rotate(rgb, angle=degree)
            depth = TF.rotate(depth, angle=degree)
            gt = TF.rotate(gt, angle=degree)

            # Color jitter
            brightness = np.random.uniform(0.6, 1.4)
            contrast = np.random.uniform(0.6, 1.4)
            saturation = np.random.uniform(0.6, 1.4)

            rgb = TF.adjust_brightness(rgb, brightness)
            rgb = TF.adjust_contrast(rgb, contrast)
            rgb = TF.adjust_saturation(rgb, saturation)

            # Resize
            rgb = TF.resize(rgb, scale, Image.BICUBIC)
            depth = TF.resize(depth, scale, Image.NEAREST)
            gt = TF.resize(gt, scale, Image.NEAREST)
            K[0] = K[0] * _scale
            K[1] = K[1] * _scale
            K[2] = K[2] * _scale
            K[3] = K[3] * _scale

            # Crop
            width, height = rgb.size

            assert self.height <= height and self.width <= width, \
                "patch size is larger than the input size"

            h_start = random.randint(0, height - self.height)
            w_start = random.randint(0, width - self.width)

            rgb = TF.crop(rgb, h_start, w_start, self.height, self.width)
            depth = TF.crop(depth, h_start, w_start, self.height, self.width)
            gt = TF.crop(gt, h_start, w_start, self.height, self.width)

            K[2] = K[2] - w_start
            K[3] = K[3] - h_start

            rgb = TF.to_tensor(rgb)
            # rgb = TF.normalize(rgb, (0.485, 0.456, 0.406),
            #                    (0.229, 0.224, 0.225), inplace=True)

            depth = TF.to_tensor(np.array(depth))
            depth = depth / _scale

            gt = TF.to_tensor(np.array(gt))
            gt = gt / _scale
        elif self.mode in ['train']:
            # Top crop if needed
            if settings.top_crop > 0:
                width, height = rgb.size
                rgb = TF.crop(rgb, settings.top_crop.top_crop, 0,
                              height - settings.top_crop, width)
                depth = TF.crop(depth, settings.top_crop, 0,
                                height - settings.top_crop, width)
                gt = TF.crop(gt, settings.top_crop, 0,
                             height - settings.top_crop, width)
                K[3] = K[3] - settings.top_crop

            # Crop
            width, height = rgb.size

            assert self.height <= height and self.width <= width, \
                "patch size is larger than the input size"

            h_start = random.randint(0, height - self.height)
            w_start = random.randint(0, width - self.width)

            rgb = TF.crop(rgb, h_start, w_start, self.height, self.width)
            depth = TF.crop(depth, h_start, w_start, self.height, self.width)
            gt = TF.crop(gt, h_start, w_start, self.height, self.width)

            K[2] = K[2] - w_start
            K[3] = K[3] - h_start

            rgb = TF.to_tensor(rgb)
            # rgb = TF.normalize(rgb, (0.485, 0.456, 0.406),
            #                    (0.229, 0.224, 0.225), inplace=True)

            depth = TF.to_tensor(np.array(depth))
            gt = TF.to_tensor(np.array(gt))
        else:
            if settings.top_crop > 0:
                width, height = rgb.size
                rgb = TF.crop(rgb, settings.top_crop, 0,
                              height - settings.top_crop, width)
                depth = TF.crop(depth, settings.top_crop, 0,
                                height - settings.top_crop, width)

                gt = TF.crop(gt, settings.top_crop, 0,
                             height - settings.top_crop, width)
                K[3] = K[3] - settings.top_crop

            rgb = TF.to_tensor(rgb)
            # rgb = TF.normalize(rgb, (0.485, 0.456, 0.406),
            #                    (0.229, 0.224, 0.225), inplace=True)

            depth = TF.to_tensor(np.array(depth))
            gt = TF.to_tensor(np.array(gt))
        if settings.num_sample > 0:
            depth = self.get_sparse_depth(depth, settings.num_sample)

        output = {'rgb': rgb, 'dep': depth, 'gt': gt, 'K': torch.Tensor(K)}
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

    def _load_data(self, idx):
        if self.mode == "train":
            path = self.sample_list[idx].split(" ")
            path_rgb = os.path.join(settings.KITTI_PATH,
                                    self.rgb_path.format( path[0], path[1], path[2]))
            path_depth = os.path.join(settings.KITTI_PATH, self.depth_path.format(path[0], path[1], path[2]))
            path_gt = os.path.join(settings.KITTI_PATH, self.gt_path.format(path[0], path[1], path[2]))
            path_calib = os.path.join(settings.KITTI_PATH, self.K_path.format(path[0][0:10],path[0][0:10]))
        elif self.mode == "val":
            path = self.sample_list[idx].split(" ")
            path_rgb = os.path.join(settings.KITTI_PATH,self.rgb_path.format(path[0], path[1]))
            path_depth = os.path.join(settings.KITTI_PATH, self.depth_path.format(path[0], path[1]))
            path_gt = os.path.join(settings.KITTI_PATH, self.gt_path.format(path[0], path[1]))
            path_calib = os.path.join(settings.KITTI_PATH, self.K_path.format(path[0], path[1]))
        else:
            path = self.sample_list[idx]
            path_rgb = os.path.join(settings.KITTI_PATH, self.rgb_path.format(path))
            path_depth = os.path.join(settings.KITTI_PATH, self.depth_path.format(path))
            path_gt = os.path.join(settings.KITTI_PATH, self.gt_path.format(path))
            path_calib = os.path.join(settings.KITTI_PATH, self.K_path.format(path))

        if self.mode in ['train']:
            calib = read_calib_file(path_calib)
            if 'image_02' in path_rgb:
                K_cam = np.reshape(calib['P_rect_02'], (3, 4))
            elif 'image_03' in path_rgb:
                K_cam = np.reshape(calib['P_rect_03'], (3, 4))
            K = [K_cam[0, 0], K_cam[1, 1], K_cam[0, 2], K_cam[1, 2]]
        else:
            f_calib = open(path_calib, 'r')
            K_cam = f_calib.readline().split(' ')
            f_calib.close()
            K = [float(K_cam[0]), float(K_cam[4]), float(K_cam[2]),
                 float(K_cam[5])]

        depth = read_depth(path_depth)

        gt = read_depth(path_gt)
        # print(path_rgb)
        rgb = Image.open(path_rgb)

        depth = Image.fromarray(depth.astype('float32'), mode='F')

        gt = Image.fromarray(gt.astype('float32'), mode='F')

        w1, h1 = rgb.size
        w2, h2 = depth.size
        w3, h3 = gt.size
        assert w1 == w2 and w1 == w3 and h1 == h2 and h1 == h3

        return rgb, depth, gt, K

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

    # A workaround for a pytorch bug
    # https://github.com/pytorch/vision/issues/2194
    class ToNumpy:
        def __call__(self, sample):
            return np.array(sample)


if __name__ == '__main__':
    print(" ")
