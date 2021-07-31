import cv2
import numpy as np
import os
import sys

import torch
import torch.utils.data as tData


class singleDataset(tData.Dataset):
    def __init__(self, hr_path='./DIV2K/DIV2K_train_HR', lr_path='./DIV2K_train_LR_bicubic/X2',
                 split='train', patch_width=128, patch_height=128, repeat=1, aug_mode=True, value_range=255.0, scale=2):
        self.hr_path = hr_path
        self.lr_path = lr_path
        self.split = split
        self.patch_width, self.patch_height = patch_width, patch_height
        self.repeat = repeat
        self.aug_mode = aug_mode
        self.value_range = value_range
        self.scale = scale

        self._load_data()

    def _load_data(self):
        self.hr_list = os.listdir(self.hr_path)
        self.data_len = len(self.hr_list)
        self.full_len = self.data_len * self.repeat

    def __len__(self):
        return self.full_len

    def __getitem__(self, index):
        idx = index % self.data_len
        img_name = self.hr_list[idx]
        key_name, img_type = img_name.split('.')

        url_hr = os.path.join(self.hr_path, '{}.{}'.format(key_name, img_type))
        url_lr = os.path.join(self.lr_path, '{}x{}.{}'.format(key_name, self.scale, img_type))

        img_hr = cv2.imread(url_hr, cv2.IMREAD_COLOR)
        img_lr = cv2.imread(url_lr, cv2.IMREAD_COLOR)

        h, w = img_lr.shape[:2]

        if self.split == 'train':
            x = np.random.randint(0, w - self.patch_width + 1)
            y = np.random.randint(0, h - self.patch_height + 1)
            img_lr = img_lr[y: y + self.patch_height, x: x + self.patch_width, :]
            img_hr = img_hr[y * self.scale: y * self.scale + self.patch_height * self.scale,
                            x * self.scale: x * self.scale + self.patch_width * self.scale, :]

        if self.aug_mode:
            # horizontal flip
            if np.random.random() > 0.5:
                img_lr = img_lr[:, ::-1, :]
                img_hr = img_hr[:, ::-1, :]
            # vertical flip
            if np.random.random() > 0.5:
                img_lr = img_lr[::-1, :, ]
                img_hr = img_hr[::-1, :, ]
            # rotate
            if np.random.random() > 0.5:
                img_lr = img_lr.transpose(1, 0, 2)
                img_hr = img_hr.transpose(1, 0, 2)

        img_lr = np.transpose(img_lr[:, :, ::-1], (2, 0, 1)).astype(np.float32) / self.value_range
        img_hr = np.transpose(img_hr[:, :, ::-1], (2, 0, 1)).astype(np.float32) / self.value_range
        img_lr = torch.from_numpy(img_lr).float()
        img_hr = torch.from_numpy(img_hr).float()

        return img_lr, img_hr


if __name__ == '__main__':
    D = singleDataset(hr_path='/data/qilu/projects/vincent/datasets/DIV2K/DIV2K_train_HR',
                    lr_path='/data/qilu/projects/vincent/datasets/DIV2K/DIV2K_train_LR_bicubic/X2')
    print(D.data_len, D.full_len)
    lr, hr = D.__getitem__(5)
    print(lr.size(), hr.size())
    print('Done')
