"""
The dataloading and processing units.
Di Wu follows the tutorial below. (Di Wu used a lot of customized function,
which in respect may not be the optimal sollution.
http://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class
"""
from __future__ import print_function, division
import os
import random
from skimage import io, transform
from scipy.misc import imresize
import numpy as np

from PIL import Image
import cv2 as cv

import torch
from torch.utils.data import Dataset, DataLoader
from code_base.tools.PyTorch_model_training import prepare_data_image_list

import matplotlib
# matplotlib.use('TkAgg')
# from matplotlib import pyplot as plt
# from torchvision import transforms as T
# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


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
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = transform.resize(image, (new_h, new_w))
        label = imresize(label, size=(new_h, new_w), interp='nearest', mode='F')

        return {'image': image, 'label': label}


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
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        label = label[top: top + new_h,
                left: left + new_w]

        return {'image': image, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label)}


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, image, label):
        if random.random() < 0.5:
            results = [image.transpose(Image.FLIP_LEFT_RIGHT),
                       label.transpose(Image.FLIP_LEFT_RIGHT)]
        else:
            results = [image, label]
        return results


class Dataset_Generators_Synthia():
    """ Initially we use synthia dataset"""

    def __init__(self, cf):
        self.cf = cf
        # Load training set
        print('\n > Loading training, valid, test set')
        train_dataset = ImageDataGenerator_Synthia(cf.dataset_path, 'train', cf=cf, crop=True, flip=True)
        val_dataset = ImageDataGenerator_Synthia(cf.dataset_path, 'valid', cf=cf, crop=False, flip=False)
        self.train_loader = DataLoader(train_dataset, batch_size=cf.batch_size, shuffle=True, num_workers=cf.workers, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, batch_size=1, num_workers=cf.workers, pin_memory=True)


class ImageDataGenerator_Synthia(Dataset):
    def __init__(self, root_dir, dataset_split, cf, crop=True, flip=True):
        """
        :param root_dir: Directory will all the images
        :param label_dir: Directory will all the label images
        :param transform:  (callable, optional): Optional tra
        nsform to be applied
        """
        self.root_dir = root_dir
        # with open(os.path.join(root_dir, 'ALL.txt')) as text_file:  # can throw FileNotFoundError
        #     lines = tuple(l.split() for l in text_file.readlines())
        self.image_dir = os.path.join(root_dir, 'RGB')
        self.label_dir = os.path.join(root_dir, 'GTTXT')
        image_files = sorted(os.listdir(self.image_dir))
        # if img_name not in ['ap_000_02-11-2015_18-02-19_000062_3_Rand_2.png',
        #                     'ap_000_02-11-2015_18-02-19_000129_2_Rand_16.png',
        #                     'ap_000_01-11-2015_19-20-57_000008_1_Rand_0.png']
        train_num = int(len(image_files) * cf.train_ratio)
        if dataset_split == 'train':
            self.image_files = image_files[:train_num]
            self.image_num = train_num
            print('Total training number is: %d'%train_num)
        elif dataset_split == 'valid':
            self.image_files = image_files[train_num:]
            self.image_num = len(image_files) - train_num
            print('Total valid number is: %d' % self.image_num)
        self.crop = crop
        self.crop_size = cf.crop_size
        self.flip = flip
        self.mean = cf.rgb_mean
        self.std = cf.rgb_std
        self.ignore_index = cf.ignore_index

    def __len__(self):
        return self.image_num

    def __getitem__(self, item):
        # Load images and perform augmentations with PIL
        img_name = os.path.join(self.image_dir, self.image_files[item])

        try:
            input = Image.open(img_name)
        except IOError:
            # unfortunately, some images are corrupted. Hence, we need to manually exclude them.
            print("Image failed loading: ", img_name)

        label_name = os.path.join(self.label_dir, self.image_files[item][:-4] + '.txt')
        try:
            with open(label_name) as text_file:  # can throw FileNotFoundError
                lines = tuple(l.split() for l in text_file.readlines())
        except IOError:
            # unfortunately, some images are corrupted. Hence, we need to manually exclude them.
            print("Label failed loading: ", img_name)

        target = np.asarray(lines).astype('int32')
        target[target == -1] = 0
        target[target == 0] = self.ignore_index
        target = np.array(target).astype('uint8')
        target = Image.fromarray(target)

        # Random uniform crop
        if self.crop:
            w, h = input.size
            x1, y1 = random.randint(0, w - self.crop_size), random.randint(0, h - self.crop_size)
            try:
                input, target = input.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size)), \
                                target.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
            except IOError:
                # unfortunately, some images are corrupted. Hence, we need to manually exclude them.
                print("image failed loading: ", img_name)
                # ap_000_01-11-2015_19-20-57_000008_1_Rand_0.png
        # Random horizontal flip
        if self.flip:
            if random.random() < 0.5:
                input, target = input.transpose(Image.FLIP_LEFT_RIGHT), target.transpose(Image.FLIP_LEFT_RIGHT)

        # Convert to tensors
        w, h = input.size
        input_t = torch.ByteTensor(torch.ByteStorage.from_buffer(input.tobytes())).view(h, w, 3).permute(2, 0, 1).float().div(255)
        target_t = torch.ByteTensor(torch.ByteStorage.from_buffer(target.tobytes())).view(h, w).long()
        # Normalise input
        input_t[0].sub_(self.mean[0]).div_(self.std[0])
        input_t[1].sub_(self.mean[1]).div_(self.std[1])
        input_t[2].sub_(self.mean[2]).div_(self.std[2])

        return input_t, target_t


class DataGenerator_Synthia_car_trajectory():
        """ Initially we use synthia dataset"""

        def __init__(self, cf):
            self.cf = cf
            print('Loading data')
            train_data, valid_data, test_data, self.data_mean, self.data_std, train_img_list, valid_img_list, test_img_list = prepare_data_image_list(cf)

            # deal images and relocate train_img_list & valid_img_list & test_img_list
            self.root_dir = '/'.join(cf.dataset_path[0].split('/')[:-1])

            print('\n > Loading training, valid, test set')
            train_dataset = Resized_BB_ImageDataGenerator_Synthia(cf, train_data, train_img_list, crop=False, flip=False)
            valid_dataset = Resized_BB_ImageDataGenerator_Synthia(cf, valid_data, valid_img_list,  crop=False, flip=False)
            test_dataset = Resized_BB_ImageDataGenerator_Synthia(cf, test_data, test_img_list, crop=False, flip=False)

            self.train_loader = DataLoader(train_dataset, batch_size=cf.batch_size_train, shuffle=True,
                                           num_workers=cf.workers, pin_memory=True)
            self.valid_loader = DataLoader(valid_dataset, batch_size=cf.batch_size_valid, num_workers=cf.workers,
                                           pin_memory=True)
            self.test_loader = DataLoader(test_dataset, batch_size=cf.batch_size_test, num_workers=cf.workers,
                                          pin_memory=True)


class BB_ImageDataGenerator_Synthia(Dataset):
    def __init__(self, cf, trajectory_data, img_list, crop=True, flip=True):
        """
        :param root_dir: Directory will all the images
        :param label_dir: Directory will all the label images
        :param transform:  (callable, optional): Optional tra
        nsform to be applied
        """
        self.trajectory_data = trajectory_data
        self.cf = cf
        self.img_list = img_list
        self.root_dir = '/'.join(cf.dataset_path[0].split('/')[:-1])

        # with open(os.path.join(root_dir, 'ALL.txt')) as text_file:  # can throw FileNotFoundError
        #     lines = tuple(l.split() for l in text_file.readlines())
        # self.image_dir = os.path.join(self.root_dir, 'RGB')
        # self.label_dir = os.path.join(self.root_dir, 'GTTXT')
        # image_files = sorted(os.listdir(self.image_dir))
        # train_num = int(len(image_files) * cf.train_ratio)
        # if dataset_split == 'train':
        #     self.image_files = image_files[:train_num]
        #     self.image_num = train_num
        #     print('Total training number is: %d'%train_num)
        # elif dataset_split == 'valid':
        #     self.image_files = image_files[train_num:]
        #     self.image_num = len(image_files) - train_num
        #     print('Total valid number is: %d' % self.image_num)
        self.crop = crop
        # self.crop_size = cf.crop_size
        self.flip = flip
        self.mean = cf.rgb_mean
        self.std = cf.rgb_std
        # self.ignore_index = cf.ignore_index

    def __len__(self):
        return len(self.trajectory_data)

    def __getitem__(self, item):

        # semantics
        img_dir = self.root_dir + '/' + self.img_list[item][0].split('/')[0] + '/' + 'GT/LABELS' + '/' + self.cf.data_stereo + '/' + self.cf.data_camera

        def img_name(i):
            return os.path.join(img_dir, self.img_list[item][i].split('/')[1])

        def semantic_image(img_name):
            try:
                input = cv.imread(img_name, -1)
                semantic_image = np.int8(input[:, :, 2])
            except IOError:
                # unfortunately, some images are corrupted. Hence, we need to manually exclude them.
                print("Image failed loading: ", img_name)

            # resize
            semantic_image = imresize(semantic_image, size=0.125, interp='nearest', mode='F')
            # Convert to training labels
            w, h = semantic_image.shape
            # Create one-hot encoding
            semantic_image_one_hot = np.zeros(shape=(self.cf.cnn_class_num, w, h))
            for c in range(self.cf.cnn_class_num):
                semantic_image_one_hot[c][semantic_image == c] = 1
            # Convert to tensors
            semantic_image_t = torch.Tensor(semantic_image_one_hot)
            return semantic_image_t

        # trajectory
        trajectory = self.trajectory_data[item]
        trajectory_t = torch.FloatTensor(trajectory)
        input_trajectorys = trajectory_t[:self.cf.lstm_input_frame, :]
        target_trajectorys = trajectory_t[self.cf.lstm_input_frame:, :]
        # semantics
        if self.cf.model_name == 'CNN_LSTM_To_FC':
            semantic_images = torch.stack([semantic_image(img_name(i)) for i in range(self.cf.lstm_input_frame)], dim=0)
        else:
            semantic_images = torch.FloatTensor(torch.zeros(input_trajectorys.size()))

        return semantic_images, input_trajectorys, target_trajectorys

# resized semantic image
class Resized_BB_ImageDataGenerator_Synthia(Dataset):
    def __init__(self, cf, trajectory_data, img_list, crop=True, flip=True):
        """
        :param root_dir: Directory will all the images
        :param label_dir: Directory will all the label images
        :param transform:  (callable, optional): Optional tra
        nsform to be applied
        """
        self.trajectory_data = trajectory_data
        self.cf = cf
        self.img_list = img_list
        self.root_dir = '/'.join(cf.dataset_path[0].split('/')[:-1])

        self.crop = crop
        # self.crop_size = cf.crop_size
        self.flip = flip
        self.mean = cf.rgb_mean
        self.std = cf.rgb_std
        # self.ignore_index = cf.ignore_index

    def __len__(self):
        return len(self.trajectory_data)

    def __getitem__(self, item):

        # trajectory
        trajectory = self.trajectory_data[item]
        trajectory_t = torch.FloatTensor(trajectory)
        input_trajectorys = trajectory_t[:self.cf.lstm_input_frame, :]
        target_trajectorys = trajectory_t[self.cf.lstm_input_frame:, :]

        # semantics
        if self.cf.model_name == 'CNN_LSTM_To_FC':
            semantic_images = self.img_list[item]
        else:
            semantic_images = torch.FloatTensor(torch.zeros(input_trajectorys.size()))

        return semantic_images, input_trajectorys, target_trajectorys


class Dataset_Generators_Cityscape():
    """ Initially we use synthia dataset"""

    def __init__(self, cf):
        self.cf = cf

        # Load training set
        print('\n > Loading training, valid, test set')
        # train_dataset = CityscapesDataset(cf=cf, split='train', transform=T.Compose([RandomCrop(cf.random_size_crop),
        #                                                                              RandomHorizontalFlip(),
        #                                                                              ToTensor(),
        #                                                                              T.Normalize(mean=cf.mean, std=cf.std)]))
        train_dataset = CityscapesDataset(cf=cf, split='train', crop=True, flip=True)
        val_dataset = CityscapesDataset(cf=cf, split='val', crop=False, flip=False)
        test_dataset = CityscapesDataset(cf=cf, split='test', crop=False, flip=False)
        self.train_loader = DataLoader(train_dataset, batch_size=cf.batch_size, shuffle=True, num_workers=cf.workers, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, batch_size=1, num_workers=cf.workers, pin_memory=True)
        self.test_loader = DataLoader(test_dataset, batch_size=cf.batch_size, num_workers=cf.workers, pin_memory=True)


class CityscapesDataset(Dataset):
    def __init__(self, cf, split='train', crop=True, flip=True):
        super().__init__()
        self.crop = crop
        self.crop_size = cf.crop_size
        self.flip = flip
        self.inputs = []
        self.targets = []
        self.num_classes = cf.num_classes
        self.full_to_train = cf.full_to_train
        self.train_to_full = cf.train_to_full
        self.full_to_colour = cf.full_to_colour
        self.mean = cf.mean
        self.std = cf.std
        #self.transform = transform

        for root, _, filenames in os.walk(os.path.join(cf.dataroot_dir, 'leftImg8bit', split)):
            for filename in filenames:
                if os.path.splitext(filename)[1] == '.png':
                    filename_base = '_'.join(filename.split('_')[:-1])
                    target_root = os.path.join(cf.dataroot_dir, 'gtFine', split, os.path.basename(root))
                    self.inputs.append(os.path.join(root, filename_base + '_leftImg8bit.png'))
                    self.targets.append(os.path.join(target_root, filename_base + '_gtFine_labelIds.png'))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        # Load images and perform augmentations with PIL
        input, target = Image.open(self.inputs[i]), Image.open(self.targets[i])
        # Random uniform crop
        if self.crop:
            w, h = input.size
            x1, y1 = random.randint(0, w - self.crop_size), random.randint(0, h - self.crop_size)
            input, target = input.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size)), \
                            target.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        # Random horizontal flip
        if self.flip:
            if random.random() < 0.5:
                input, target = input.transpose(Image.FLIP_LEFT_RIGHT), target.transpose(Image.FLIP_LEFT_RIGHT)

        # Convert to tensors
        w, h = input.size

        input = torch.ByteTensor(torch.ByteStorage.from_buffer(input.tobytes())).view(h, w, 3).permute(2, 0, 1).float().div(255)
        target = torch.ByteTensor(torch.ByteStorage.from_buffer(target.tobytes())).view(h, w).long()
        # Normalise input
        input[0].sub_(self.mean[0]).div_(self.std[0])
        input[1].sub_(self.mean[1]).div_(self.std[1])
        input[2].sub_(self.mean[2]).div_(self.std[2])
        # Convert to training labels
        target_clone = target.clone()
        for k, v in self.full_to_train.items():
            target_clone[target == k] = v
        # Create one-hot encoding
        target_one_hot = torch.zeros(self.num_classes, h, w)
        for c in range(self.num_classes):
            target_one_hot[c][target_clone == c] = 1

        # TOD): dangerous hack below
        #target_clone[target == self.num_classes] = 0
        return input, target

