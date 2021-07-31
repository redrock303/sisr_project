from easydict import EasyDict as edict

from dataset.singleDataset import singleDataset
from dataset.mixDataset import mixDataset


class DATASET:
    LEGAL = ['DIK2K', 'Flickr2K', 'Set5']

    DIK2K = edict()
    DIK2K.TRAIN = edict()
    DIK2K.TRAIN.HR = '/data1/zk/vsr/dataset/DIV2K_train_HR_sub'  # 32208
    DIK2K.TRAIN.LRx2 = '/data1/zk/vsr/dataset/DIV2K_train_LR_bicubic_sub/X2'
    DIK2K.VAL = edict()
    DIK2K.VAL.HR = '/data1/zk/vsr/dataset/DIV2K_valid_HR'
    DIK2K.VAL.LRx2 = '/data1/zk/vsr/dataset/DIV2K_valid_LR_bicubic/X2'

    Flickr2K = edict()
    Flickr2K.TRAIN = edict()
    Flickr2K.TRAIN.HR = '/data1/zk/vsr/dataset/Flickr2K/Flickr2K_HR_sub'  # 106641
    Flickr2K.TRAIN.LRx2 = '/data1/zk/vsr/dataset/Flickr2K/Flickr2K_LR_bicubic_sub/X2'

    Set5 = edict()
    Set5.VAL = dict()
    Set5.VAL.HR = '/data1/zk/vsr/dataset/testset/benchmark/Set5/HR'
    Set5.VAL.LRx2 = '/data1/zk/vsr/dataset/testset/benchmark/Set5/LR_bicubic/X2'


def get_train_dataset(config):
    hr_paths = []
    lr_paths = []
    D = DATASET()

    datasets = config.DATASET.DATASETS
    scale = config.DATASET.SCALE
    for dataset in datasets:
        if dataset not in D.LEGAL:
            raise ValueError('Illegal dataset.')
        hr_paths.append(eval('D.%s.TRAIN.HR' % dataset))
        lr_paths.append(eval('D.%s.TRAIN.LRx%d' % (dataset, scale)))

    return mixDataset(hr_paths, lr_paths, split='train', patch_width=config.DATASET.PATCH_WIDTH,
                      patch_height=config.DATASET.PATCH_HEIGHT, repeat=config.DATASET.REPEAT,
                      aug_mode=config.DATASET.AUG_MODE, value_range=config.DATASET.VALUE_RANGE, scale=scale)


def get_val_dataset(config):
    D = DATASET()

    if config.VAL.DATASET not in D.LEGAL:
        raise ValueError('Illegal dataset.')

    hr_path = eval('D.%s.VAL.HR' % config.VAL.DATASET)
    lr_path = eval('D.%s.VAL.LRx%d' % (config.VAL.DATASET, config.DATASET.SCALE))

    return singleDataset(hr_path, lr_path, split='val', patch_width=config.DATASET.PATCH_WIDTH,
                         patch_height=config.DATASET.PATCH_HEIGHT, repeat=1, aug_mode=False,
                         value_range=config.DATASET.VALUE_RANGE, scale=config.DATASET.SCALE)


if __name__ == '__main__':
    from exps.baseline.config import config

    config.DATASET.DATASETS = ['DIK2K']
    # dataset = get_val_dataset(config)
    dataset = get_train_dataset(config)
    print(dataset.full_len)
    lr, hr = dataset.__getitem__(199)
    print(lr.size(), hr.size())

