"""
The dataloading and processing units.
Di Wu follows the tutorial below. (Di Wu used a lot of customized function,
which in respect may not be the optimal sollution.
http://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class
"""
from __future__ import print_function, division
import os
from skimage import io
from scipy.misc import imresize
import numpy as np
import cv2 as cv

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, classes, instances, depth = sample['image'], sample['classes'], sample['instances'], sample['depth']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = imresize(image, size=(new_h, new_w))
        classes = imresize(classes, size=(new_h, new_w), interp='nearest', mode='F')
        instances = imresize(instances, size=(new_h, new_w), interp='nearest', mode='F')
        depth = imresize(depth, size=(new_h, new_w), interp='nearest', mode='F')
        return {'image': image, 'classes': classes, 'instances': instances, 'depth': depth}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, classes, instances, depth = sample['image'], sample['classes'], sample['instances'], sample['depth']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        classes = classes[top: top + new_h, left: left + new_w]
        instances = instances[top: top + new_h, left: left + new_w]
        depth = depth[top: top + new_h, left: left + new_w]

        return {'image': image, 'classes': classes, 'instances': instances, 'depth': depth}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, classes, instances, depth = sample['image'], sample['classes'], sample['instances'], sample['depth']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'classes': torch.from_numpy(classes),
                'instances': torch.from_numpy(instances),
                'depth': torch.from_numpy(depth)}


class Dataset_Generators_Synthia_Car_trajectory():
    """ Initially we use synthia dataset"""

    def __init__(self, cf):
        self.cf = cf
        self.dataloader = {}

        # Load training set
        print('\n > Loading training set, currently we have only training set')
        dataloaders_single = {x: ImageDataGenerator_Synthia_Car_trajectory(
            cf=cf,
            transform=T.Compose([
                Rescale(cf.resize_train),
                #RandomCrop(cf.random_size_crop),
                ToTensor(),
            ]))
            for x in ['train']}

        self.dataloader['train'] = DataLoader(dataset=dataloaders_single['train'],
                                              batch_size=cf.batch_size_train,
                                              shuffle=cf.shuffle_train,
                                              num_workers=cf.dataloader_num_workers_train)


class ImageDataGenerator_Synthia_Car_trajectory(Dataset):
    """ Image Data"""

    def __init__(self, cf, transform=None):
        """

        :param root_dir: Directory will all the images
        :param label_dir: Directory will all the label images
        :param transform:  (callable, optional): Optional transform to be applied
        """
        self.image_dir = os.path.join(cf.dataset_path, cf.data_type, cf.data_stereo, cf.data_camera)
        self.image_files = sorted(os.listdir(self.image_dir))[cf.start_tracking_idx:]
        self.label_dir = os.path.join(cf.dataset_path, 'GT', cf.data_label, cf.data_stereo, cf.data_camera)
        self.label_files = sorted(os.listdir(self.label_dir))[cf.start_tracking_idx:]
        self.depth_dir = os.path.join(cf.dataset_path, 'Depth', cf.data_stereo, cf.data_camera)
        self.depth_files = sorted(os.listdir(self.depth_dir))[cf.start_tracking_idx:]
        self.len = len(os.listdir(self.image_dir)[cf.start_tracking_idx:])
        # we need to check all the image, label, depth have the same number of files
        assert len(self.image_files) == len(self.label_files) == len(self.depth_files), "number of files are not equal"
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        img_name = os.path.join(self.image_dir, self.image_files[item])
        image = io.imread(img_name)
        label_name = os.path.join(self.label_dir, self.image_files[item])
        depth_name = os.path.join(self.depth_dir, self.image_files[item])
        # folder containing png files (one per image). Annotations are given in two channels. The first
        # channel contains the class of that pixel (see the table below). The second channel contains
        # the unique ID of the instance for those objects that are dynamic (cars, pedestrians, etc.).
        label = cv.imread(label_name, -1)
        depth = cv.imread(depth_name, -1)
        classes = np.uint8(label[:, :, 2])
        instances = np.uint8(label[:, :, 1])
        sample = {'image': image, 'classes': classes, 'instances': instances, 'depth': depth[:, :, 0].astype(np.int32)}

        if self.transform:
            sample = self.transform(sample)

        # Di Wu also save the image name here for the future documentation and
        # it could be useful for time series prediction
        sample['img_name'] = self.image_files[item]
        return sample

