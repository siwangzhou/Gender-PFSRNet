import os
import random
import numpy as np
from PIL import Image
import imgaug as ia
import imgaug.augmenters as iaa

from data.image_folder import make_dataset

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from data.base_dataset import BaseDataset
from utils.utils import onehot_parse_map
import time

from data.ffhq_dataset import random_gray, complex_imgaug

class CelebADataset(BaseDataset):

    def __init__(self, opt):
        # self.dataroot = opt.dataroot
        self.img_size = opt.Pimg_size
        self.lr_size = opt.Gin_size
        self.hr_size = opt.Gout_size
        self.shuffle = True if opt.isTrain else False

        # self.img_dataset = sorted(make_dataset(os.path.join(opt.dataroot, 'CelebA-HQ-img')))
        # self.mask_dataset = sorted(make_dataset(os.path.join(opt.dataroot, 'CelebAMask-HQ-mask-color')))
        self.img_path = os.path.join(opt.dataroot, 'CelebA-HQ-sr')
        # self.img_path = os.path.join(opt.dataroot, 'r50_1.0_8w')
        self.mask_path = os.path.join(opt.dataroot, 'CelebAMask-HQ-mask-color')
        self.attr_path = os.path.join(opt.dataroot, 'CelebAMask-HQ-attribute.txt')


        self.attr2idx = {}
        self.idx2attr = {}
        self.selected_attrs = opt.selected_attrs

        self.dataset = []

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # self.to_lrtensor = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Resize((64, 64)),
        #     transforms.Resize((256, 256)),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # ])

        self.preprocess()

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]


        for i, line in enumerate(lines):
            split = line.split()
            # filename = split[0]
            filename = split[0][:-3] + 'png'
            values = split[1:]


            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')
                # label.append(values[idx] == '0') # To Female

            self.dataset.append([os.path.join(self.img_path, filename), os.path.join(self.mask_path, filename[:-3]+'png'), label])




        print('Finished preprocessing the CelebA dataset...')
        print(len(self.dataset))

        " 打乱数据集 "
        if self.shuffle:
            random.seed(time.time())
            random.shuffle(self.dataset)


    def __len__(self, ):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path = self.dataset[idx][0]
        mask_path = self.dataset[idx][1]
        img_label = self.dataset[idx][2]
        hr_img = Image.open(img_path).convert('RGB')
        mask_img = Image.open(mask_path).convert('RGB')

        hr_img = hr_img.resize((self.hr_size, self.hr_size))
        # hr_img = random_gray(hr_img, p=0.3)
        # scale_size = np.random.randint(32, 256)
        scale_size = self.hr_size // 4
        lr_img = complex_imgaug(hr_img, self.img_size, scale_size)

        mask_img = mask_img.resize((self.hr_size, self.hr_size))
        mask_label = onehot_parse_map(mask_img)
        mask_label = torch.tensor(mask_label).float()

        hr_tensor = self.to_tensor(hr_img)
        lr_tensor = self.to_tensor(lr_img)

        return {'HR': hr_tensor, 'LR': lr_tensor, 'HR_paths': img_path, 'Mask': mask_label, 'Attr': torch.LongTensor(img_label)}
        # return {'HR': hr_tensor, 'LR': lr_tensor, 'HR_paths': img_path, 'label': torch.LongTensor(img_label)}