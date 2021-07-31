import cv2
import numpy as np
import os
import sys

import torch
import torch.utils.data as tData


class mixDataset(tData.Dataset):
    def __init__(self, hr_paths, lr_paths, split='train', patch_width=128, patch_height=128, repeat=1, aug_mode=True,
                 value_range=255.0, scale=2):
        self.hr_paths = hr_paths
        self.lr_paths = lr_paths
        self.split = split
        self.patch_width, self.patch_height = patch_width, patch_height
        self.repeat = repeat
        self.aug_mode = aug_mode
        self.value_range = value_range
        self.scale = scale

        self._load_data()

    def _load_data(self):
        assert len(self.lr_paths) == len(self.hr_paths), 'Illegal hr-lr dataset mappings.'

        self.hr_list = []
        self.lr_list = []
        for hr_path in self.hr_paths:
            hr_imgs = sorted(os.listdir(hr_path))
            for hr_img in hr_imgs:
                self.hr_list.append(os.path.join(hr_path, hr_img))
        for lr_path in self.lr_paths:
            lr_imgs = sorted(os.listdir(lr_path))
            for lr_img in lr_imgs:
                self.lr_list.append(os.path.join(lr_path, lr_img))

        assert len(self.hr_list) == len(self.lr_list), 'Illegal hr-lr mappings.'

        self.data_len = len(self.hr_list)
        self.full_len = self.data_len * self.repeat

    def __len__(self):
        return self.full_len

    def __getitem__(self, index):
        idx = index % self.data_len

        url_hr = self.hr_list[idx]
        url_lr = self.lr_list[idx]

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

        img_lr = np.transpose(img_lr[:, :, [2, 1, 0]], (2, 0, 1)).astype(np.float32) / self.value_range
        img_hr = np.transpose(img_hr[:, :, [2, 1, 0]], (2, 0, 1)).astype(np.float32) / self.value_range
        img_lr = torch.from_numpy(img_lr).float()
        img_hr = torch.from_numpy(img_hr).float()

        return img_lr, img_hr


if __name__ == '__main__':
    D = mixDataset(hr_paths=['/data/qilu/projects/vincent/datasets/DIV2K/DIV2K_train_HR'],
                   lr_paths=['/data/qilu/projects/vincent/datasets/DIV2K/DIV2K_train_LR_bicubic/X2'])
    print(D.data_len, D.full_len)
    lr, hr = D.__getitem__(5)
    print(lr.size(), hr.size())
    print('Done')
