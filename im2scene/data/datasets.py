import os
import logging
from torch.utils import data
import torch
import numpy as np
import glob
from PIL import Image
from torchvision import transforms
import lmdb
import pickle
import string
import io
import random
# fix for broken images
from PIL import ImageFile
import cv2
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


class ImagesDataset_afhq(data.Dataset):
    ''' Default Image Dataset Class.

    Args:
        dataset_folder (str): path to LSUN dataset
        size (int): image output size
        celebA_center_crop (bool): whether to apply the center
            cropping for the celebA and celebA-HQ datasets.
        random_crop (bool): whether to perform random cropping
        use_tanh_range (bool): whether to rescale images to [-1, 1]
    '''

    def __init__(self, dataset_folder, size=512, celebA_center_crop=False,
                 random_crop=False, use_tanh_range=False):

        self.size = size
        
        assert(not(celebA_center_crop and random_crop))
        if random_crop:
            self.transform = [
                transforms.Resize(size),
                transforms.RandomCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        elif celebA_center_crop:
            if size <= 128:  # celebA
                crop_size = 108
            else:  # celebAHQ
                crop_size = 650
            self.transform = [
                transforms.CenterCrop(crop_size),
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]
        else:
            self.transform = [
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        if use_tanh_range:
            self.transform += [
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(self.transform)

        self.data_type = os.path.basename(dataset_folder).split(".")[-1]
        assert(self.data_type in ["jpg", "png", "npy"])

        import time
        t0 = time.time()
        print('Start loading file addresses ...')
        images_cat = sorted(glob.glob('/data/afhq/'+'*/cat/*.jpg'))
        images_dog = sorted(glob.glob('/data/afhq/'+'*/dog/*.jpg'))
        images_dots = sorted(glob.glob('/data/afhq/'+'*/dots/*.jpg'))
        images_fox = sorted(glob.glob('/data/afhq/'+'*/fox/*.jpg'))
        images_lion = sorted(glob.glob('/data/afhq/'+'*/lion/*.jpg'))
        images_tiger  = sorted(glob.glob('/data/afhq/'+'*/tiger/*.jpg'))
        images_wolf = sorted(glob.glob('/data/afhq/'+'*/wolf/*.jpg'))
        self.images = images_cat + images_dog + images_dots + images_fox + images_lion + images_tiger + images_wolf
        self.dataset_label = [0] * len(images_cat) + [1] *len(images_dog) + [2] * len(images_dots) + [3] * len(images_fox) + [4] * len(images_lion) + [5] * len(images_tiger) + [6] *len(images_wolf)
        

        t = time.time() - t0
        print('done! time:', t)
        print("Number of images found: %d" % len(self.images))

        self.length = len(self.images)

    def __getitem__(self, idx):
        try:
            buf = self.images[idx]
            if self.data_type == 'npy':
                img = np.load(buf)[0].transpose(1, 2, 0)
                img = Image.fromarray(img).convert("RGB")
            else:
                img = Image.open(buf).convert('RGB')

            if self.transform is not None:
                img = self.transform(img)
            
            #added
            label = self.dataset_label[idx]
            label = torch.tensor(label)
            label = torch.nn.functional.one_hot(label, num_classes = 7)
            label = label.float()
            data = {
                'image': img,
                'cond': label
            }
            return data
        except Exception as e:
            print(e)
            print("Warning: Error occurred when loading file %s" % buf)
            return self.__getitem__(np.random.randint(self.length))

    def __len__(self):
        return self.length
    


class ImagesDataset(data.Dataset):
    ''' Default Image Dataset Class.

    Args:
        dataset_folder (str): path to LSUN dataset
        size (int): image output size
        celebA_center_crop (bool): whether to apply the center
            cropping for the celebA and celebA-HQ datasets.
        random_crop (bool): whether to perform random cropping
        use_tanh_range (bool): whether to rescale images to [-1, 1]
    '''

    def __init__(self, dataset_folder, attr_path, size=64, celebA_center_crop=False,
                 random_crop=False, use_tanh_range=False):

        self.size = size
        self.attr2idx = {}
        self.idx2attr = {}
        self.dataset_label = []
        #self.selected_attrs = ['Eyeglasses', 'Black_Hair', 'Bangs', 'Bald']
        #self.selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']
        self.attr_dir = attr_path
        assert(not(celebA_center_crop and random_crop))
        if random_crop:
            self.transform = [
                transforms.Resize(size),
                transforms.RandomCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        elif celebA_center_crop:
            if size <= 128:  # celebA
                crop_size = 108
            else:  # celebAHQ
                crop_size = 650
            self.transform = [
                transforms.CenterCrop(crop_size),
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]
        else:
            self.transform = [
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        if use_tanh_range:
            self.transform += [
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(self.transform)

        self.data_type = os.path.basename(dataset_folder).split(".")[-1]
        assert(self.data_type in ["jpg", "png", "npy"])

        import time
        t0 = time.time()
        print('Start loading file addresses ...')
        images = sorted(glob.glob(dataset_folder))
        self.preprocess()
        t = time.time() - t0
        print('done! time:', t)
        print("Number of images found: %d" % len(images))

        self.images = images
        self.length = len(images)

    def __getitem__(self, idx):
        try:
            buf = self.images[idx]
            if self.data_type == 'npy':
                img = np.load(buf)[0].transpose(1, 2, 0)
                img = Image.fromarray(img).convert("RGB")
            else:
                img = Image.open(buf).convert('RGB')

            if self.transform is not None:
                img = self.transform(img)
            
            #added
            filename, label = self.dataset_label[idx]
            label = torch.tensor(label).float()
            data = {
                'image': img,
                'cond': label
            }
            return data
        except Exception as e:
            print(e)
            print("Warning: Error occurred when loading file %s" % buf)
            return self.__getitem__(np.random.randint(self.length))

    def __len__(self):
        return self.length
    
    def preprocess(self):
        lines = [line.rstrip() for line in open(self.attr_dir, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name
        lines = lines[2:]
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]
            label = []
            for attr_name in all_attr_names:
                idx = self.attr2idx[attr_name]
                label.append(int(values[idx] == '1'))
            self.dataset_label.append([filename, label])