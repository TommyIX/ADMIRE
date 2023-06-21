'''
ADMIRE dataset.py
王锦宏

12.22更新记录：换用了新版contour数据集，ADMTorch从此之后不再涉及mask数据
'''

import os
import cv2
import torch
import imageio
import numpy as np

from skimage.transform import resize
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from process.getGT import getGT_single

class ADMDataset(Dataset):
    def __init__(self, dir, imgsize, mode, status, imnum=1, preload_mode=False, ff_fold_num=0, ff_random_seed=0, L_Points=None):
        super(ADMDataset, self).__init__()
        assert mode in ['old', 'ff-seq', 'ff-rad', 'no']
        if mode!='no': assert status in ['train', 'test']
        if not preload_mode: assert L_Points != None

        self.imgsize = imgsize
        self.imnum = imnum
        self.image = []
        self.contour = []

        if preload_mode:  # 从预先保存的numpy文件中读取数据集
            imageset = np.load(dir[0])
            contourset = np.load(dir[1])
            self.imnum = imageset.shape[3]
            for i in range(0, self.imnum):
                self.image.append(imageset[:,:,:,i])  # 读入所有图像
                self.contour.append(contourset[:,:,i])  # 读入所有掩膜
        else:  # 从图像文件夹中逐个读取，并提取轮廓（使用新版getGT函数）
            for i in range(0, self.imnum):
                this_im = imageio.imread(os.path.join(dir[i], + 'image_' + str(i) + '.jpg')).astype(np.float32)
                this_im = resize(this_im, [imgsize, imgsize])
                self.image.append(np.float32(this_im) / 255)  # 归一化到0~1之间。

                img_mask = imageio.imread(os.path.join(dir[i], 'image_mask_' + str(i) + '.jpg')).astype(np.float32)
                img_mask = resize(img_mask, [imgsize, imgsize])
                img_mask = cv2.cvtColor(img_mask, cv2.COLOR_RGB2GRAY)
                imgcontour = getGT_single(img_mask, L_Points)
                self.contour.append(imgcontour)

        # 按照数据集载入的mode，划分数据集
        if mode == 'old':  # 相当于顺序五折的最后一折
            divide_point = int(self.imnum * 0.8)
            if status == 'train':
                self.image = self.image[:divide_point]
                self.contour = self.contour[:divide_point]
            else:
                self.image = self.image[divide_point:]
                self.contour = self.contour[divide_point:]
        elif 'ff' in mode:  # 五折交叉验证
            if mode == 'ff-rad':  # 随机五折情况下就打乱数据集后，再按顺序五折划分
                np.random.seed(ff_random_seed)
                np.random.shuffle(self.image)
                np.random.seed(ff_random_seed)
                np.random.shuffle(self.contour)
            # 下面，顺序五折中的第一折使用五部分中的第一部分作为测试集，第二折使用五部分中的第二部分作为测试集，以此类推
            fold_imnum = int(self.imnum / 5)
            if status == 'train':
                if (ff_fold_num+1)*fold_imnum<len(self.image):
                    del self.image[ff_fold_num*fold_imnum:(ff_fold_num+1)*fold_imnum]
                    del self.contour[ff_fold_num*fold_imnum:(ff_fold_num+1)*fold_imnum]
                else:
                    del self.image[ff_fold_num*fold_imnum:]
                    del self.contour[ff_fold_num*fold_imnum:]
                # 以上，从数据集和掩膜中删掉作为测试集中的那些数据，剩下的就是训练集。
                #     里面的if-else是考虑是否是最后一折的，担心最后一折剩下的数目不够导致报错，才这样写的。
            else:
                if (ff_fold_num+1)*fold_imnum<len(self.image):
                    self.image = self.image[ff_fold_num * fold_imnum:(ff_fold_num + 1) * fold_imnum]
                    self.contour = self.contour[ff_fold_num * fold_imnum:(ff_fold_num + 1) * fold_imnum]
                else:
                    self.image = self.image[ff_fold_num * fold_imnum:]
                    self.contour = self.contour[ff_fold_num * fold_imnum:]
                # 以上，构建测试集。
        # 其他情况（mode==no的情况）下不划分测试集，返回即可

    def __getitem__(self, index):
        # 因为torch.nn处理格式为[channel, imx, imy]，因此对图像进行一下轴替换
        im = np.transpose(self.image[index], [2, 0, 1])
        return torch.from_numpy(im).float(), torch.from_numpy(self.contour[index]).float()

    def __add__(self, other):
        self.image += other.image
        self.contour += other.contour
        self.imnum += other.imnum
        return self

    def __len__(self):
        return len(self.image)

    def instant_shuffle(self, seed):
        np.random.seed(seed)
        np.random.shuffle(self.image)
        np.random.seed(seed)
        np.random.shuffle(self.contour)

def build_dataloader(dir, img_size, mode, batch_size, preload_mode=False, imnum=1, ff_fold_num=0, ff_random_seed=0, L_Points=None):
    if mode == 'no':
        dataset_train = ADMDataset(dir, img_size, mode, 'all_train', imnum, preload_mode, ff_fold_num, ff_random_seed, L_Points)
        loader_train = DataLoader(dataset_train, batch_size=batch_size, num_workers=0)
        return dataset_train, None, loader_train, None
    else:
        dataset_train = ADMDataset(dir, img_size, mode, 'train', imnum, preload_mode, ff_fold_num, ff_random_seed, L_Points)
        dataset_test = ADMDataset(dir, img_size, mode, 'test', imnum, preload_mode, ff_fold_num, ff_random_seed, L_Points)
        loader_train = DataLoader(dataset_train, batch_size=batch_size, num_workers=0)
        loader_test = DataLoader(dataset_test, batch_size=batch_size, num_workers=0)
        return dataset_train, dataset_test, loader_train, loader_test

def build_dataloader_alldataset(alldirs, img_size, mode, batch_size, preload_mode=False, imnum=1, ff_fold_num=0, ff_random_seed=0, L_Points=None, instant_shuffle=False):
    datasets_train = []
    datasets_test = []
    for dir in alldirs:
        if mode == 'no':
            dataset_train = ADMDataset(dir, img_size, mode, 'all_train', imnum, preload_mode, ff_fold_num, ff_random_seed,
                                       L_Points)
            datasets_train.append(dataset_train)
        else:
            dataset_train = ADMDataset(dir, img_size, mode, 'train', imnum, preload_mode, ff_fold_num, ff_random_seed,
                                       L_Points)
            dataset_test = ADMDataset(dir, img_size, mode, 'test', imnum, preload_mode, ff_fold_num, ff_random_seed,
                                      L_Points)
            datasets_train.append(dataset_train)
            datasets_test.append(dataset_test)

    fulldataset_train = datasets_train[0]
    for i in range(1, len(datasets_train)):
        fulldataset_train += datasets_train[i]
    if instant_shuffle:
        fulldataset_train.instant_shuffle(ff_random_seed)
    fullloader_train = DataLoader(fulldataset_train, batch_size=batch_size, num_workers=0)

    if mode != 'no':
        fulldataset_test = datasets_test[0]
        for i in range(1, len(datasets_test)):
            fulldataset_test += datasets_test[i]
        if instant_shuffle:
            fulldataset_test.instant_shuffle(ff_random_seed)
        fulloader_test = DataLoader(fulldataset_test, batch_size=batch_size, num_workers=0)

    if mode=='no':
        return fulldataset_train, None, fullloader_train, None
    else:
        return fulldataset_train, fulldataset_test, fullloader_train, fulloader_test