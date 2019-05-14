import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from glob import glob
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchsummary import summary
import torch.nn.functional as F
import torch.utils.data as utils
from os import listdir
from os.path import isfile, join
import pandas as pd
from generate_roi_dataset import *
from sklearn.utils import shuffle
import operator
import cv2
from albumentations import (
    Compose,
    HorizontalFlip,
    VerticalFlip,
    RandomRotate90,
    Transpose,
    ShiftScaleRotate,
    RandomBrightness,
    RandomContrast,
    RGBShift,
    Normalize,
    Rotate
)


def cropND(img, bounding):
    """Upscaling image to size which is set as bounding"""
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]


class Dataset32(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        image = self.data.iloc[index]
        y_class = int(image['class'])
        if y_class == 1:
            p = '/datagrid/temporary/dobkomar/positive_data/train/'
            lung_img = np.load(p + image['name'])
            if lung_img.shape[0] != 32:
                lung_img = cv2.resize(lung_img, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
            X = lung_img.reshape((32, 32))

        elif y_class == 0:
            p = '/datagrid/temporary/dobkomar/output_path/0/'
            lung_img = np.load(p + image['name'])
            X = lung_img[24, :, :].reshape((48, 48))
            X = cropND(X, (32, 32))

        return X.reshape((1, 32, 32)).astype(np.float), y_class

    def __len__(self):
        return self.data.shape[0]


class DatasetD1VGG_pretrained(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        image = self.data.iloc[index]
        y_class = int(image['class'])
        lung_img = np.load(image['filename'])
        X = lung_img[16, :, :].reshape((32, 32)).astype(np.float)
        X = torch.from_numpy(X.reshape((1,32, 32)))
        X = torch.cat([X, X, X])
        return X.reshape((3, 32, 32)), y_class

    def __len__(self):
        return self.data.shape[0]


class LUNA_Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        image = self.data.iloc[index]
        y_class = int(image['class'])
        lung_img = np.load(image['filename'])
        X = lung_img[16, :, :].reshape((32, 32))
        return X.reshape((1, 32, 32)), y_class

    def __len__(self):
        return self.data.shape[0]

    
class LUNA_Dataset_3D(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        image = self.data.iloc[index]
        y_class = int(image['class'])
        lung_img = np.load(image['filename'])
        X = lung_img.reshape((32, 32, 32))
        return X.reshape((1, 32, 32, 32)), y_class

    def __len__(self):
        return self.data.shape[0]    
   
    
class NPYdataset_augmented(Dataset):
    def __init__(self, data):
        self.data = data
        self.augmentator = Compose([
            # Non destructive transformations
            VerticalFlip(p=0.6),
            HorizontalFlip(p=0.6),
            RandomRotate90(),
            Transpose(p=0.6),
            ShiftScaleRotate(p=0.2, scale_limit=(0.1, 0.3)),
            Rotate(p=0.6),
        ])
        self.normalizer = Compose([Normalize(mean=-631, std=380)])

    def __getitem__(self, index):
        image = self.data.iloc[index]
        y_class = int(image['class'])
        lung_img = np.load('/datagrid/temporary/dobkomar/luna/subset{}/'.format(image['subset']) + image['name'])
        X = lung_img[16, :, :].reshape((32, 32))
        if y_class == 1:
            augmented = self.augmentator(image=X)
            X = augmented['image']

        return X.reshape((1, 32, 32)), y_class

    def __len__(self):
        return self.data.shape[0]



class NPYdataset_3d_augmented(Dataset):
    def __init__(self, data):
        self.data = data
        self.augmentator = Compose([
            # Non destructive transformations
            VerticalFlip(p=0.6),
            HorizontalFlip(p=0.6),
            RandomRotate90(),
            Transpose(p=0.6),
            ShiftScaleRotate(p=0.2, scale_limit=(0.1, 0.3)),
            Rotate(p=0.6),
        ])
        self.normalizer = Compose([Normalize(mean=-631, std=380)])

    def __getitem__(self, index):
        image = self.data.iloc[index]
        y_class = int(image['class'])
        lung_img = np.load('/datagrid/temporary/dobkomar/luna/subset{}/'.format(image['subset']) + image['name'])
        X = lung_img.reshape((32, 32, 32))
        if y_class == 1:
            augmented = self.augmentator(image=X)
            X = augmented['image']

        return X.reshape((1, 32, 32, 32)), y_class

    def __len__(self):
        return self.data.shape[0]



class LunaDatasetRaw(Dataset):
    def __init__(self, width_size, indices):
        self.data = pd.read_csv('./data/candidates_samples.csv').ix[indices]
        self.width_size = width_size
        self.std = 377
        self.mean = -614

    def __getitem__(self, index):
        image = self.data.iloc[index]
        vox_coords = []
        for i in image['voxcoord'].split(' '):
            try:
                i = ''.join(i.split('['))
                vox_coords.append(int(i.split('.')[0]))
            except:
                pass

        y_class = int(image['class'])
        if y_class == 1:
            if self.width_size == 32:
                im_path = '/datagrid/temporary/dobkomar/dataset/subset{}/'.format(
                    image['subset']) + '{}_{}_{}.npy'.format(image['seriesuid'], image['sliceid'], image['specialid'])
            elif self.width_size == 64:
                im_path = '/datagrid/temporary/dobkomar/dataset64/subset{}/'.format(
                    image['subset']) + '{}_{}_{}.npy'.format(image['seriesuid'], image['sliceid'], image['specialid'])
            lung_img = np.load(im_path)

        elif y_class == 0:
            if self.width_size == 32:
                im_path = '/datagrid/temporary/dobkomar/dataset/subset{}/'.format(image['subset']) + '{}_{}.npy'.format(
                    image['seriesuid'], image['specialid'])
            elif self.width_size == 64:
                im_path = '/datagrid/temporary/dobkomar/dataset64/subset{}/'.format(image['subset']) + '{}_{}.npy'.format(
                    image['seriesuid'], image['specialid'])
            lung_img = np.load(im_path)

        # norm_img = ((lung_img * self.std) + self.mean)    # normalize from [-1, 1]
        norm_img = (lung_img - self.mean) / self.std      # normalize from [0, 1]

        return norm_img, y_class  # lung_img, y_class

    def __len__(self):
        return self.data.shape[0]
