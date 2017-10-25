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

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
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
        self.dataloader = {}

        # Load training set
        print('\n > Loading training, valid, test set')
        dataloaders_single = {x: ImageDataGenerator_Synthia(
            root_dir=os.path.join(cf.dataset_path, x),
            transform=T.Compose([
                Rescale(cf.resize_train),
                RandomCrop(cf.random_size_crop),
                ToTensor(),
            ]))
            for x in ['train', 'valid', 'test']}

        self.dataloader['train'] = DataLoader(dataset=dataloaders_single['train'],
                                              batch_size=cf.batch_size_train,
                                              shuffle=cf.shuffle_train,
                                              num_workers=cf.dataloader_num_workers_train)
        self.dataloader['valid'] = DataLoader(dataset=dataloaders_single['valid'],
                                              batch_size=cf.batch_size_valid,
                                              shuffle=cf.shuffle_valid,
                                              num_workers=cf.dataloader_num_workers_valid)
        self.dataloader['test'] = DataLoader(dataset=dataloaders_single['test'],
                                             batch_size=cf.batch_size_test,
                                             shuffle=cf.shuffle_test,
                                             num_workers=cf.dataloader_num_workers_test)


class ImageDataGenerator_Synthia(Dataset):
    """ Image Data"""

    def __init__(self, root_dir, transform=None):
        """

        :param root_dir: Directory will all the images
        :param label_dir: Directory will all the label images
        :param transform:  (callable, optional): Optional tra
        nsform to be applied
        """
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.image_files = os.listdir(self.image_dir)
        self.label_dir = os.path.join(root_dir, 'masks')
        self.label_files = os.listdir(self.label_dir)
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.image_dir))

    def __getitem__(self, item):
        img_name = os.path.join(self.image_dir, self.image_files[item])
        image = io.imread(img_name)
        label_name = os.path.join(self.label_dir, self.image_files[item][:-4] + '.txt')

        with open(label_name) as text_file:  # can throw FileNotFoundError
            lines = tuple(l.split() for l in text_file.readlines())

        label = np.asarray(lines).astype('int32')
        # y = imresize(y.astype('uint8'), size=x.shape[:2], interp='nearest')

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


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
        val_dataset = CityscapesDataset(cf=cf, split='val', crop=True, flip=True)
        test_dataset = CityscapesDataset(cf=cf, split='test', crop=False, flip=False)
        self.train_loader = DataLoader(train_dataset, batch_size=cf.batch_size, shuffle=True, num_workers=cf.workers, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, batch_size=cf.batch_size, num_workers=cf.workers, pin_memory=True)
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
        input[0].add_(self.mean[0]).div_(self.std[0])
        input[1].add_(self.mean[1]).div_(self.std[1])
        input[2].add_(self.mean[2]).div_(self.std[2])
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
        return input, target_one_hot, target_clone  # Return x, y (one-hot), y (index)

