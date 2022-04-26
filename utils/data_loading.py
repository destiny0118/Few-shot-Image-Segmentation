import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from torchvision import  transforms
from matplotlib import pyplot as plt

class CarvanaDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = '_mask',augment:bool=False,aug_dir:str='',aug_op:str=''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.aug_dir=Path(aug_dir)
        '''是否进行增强操作'''
        self.augment=augment
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        '''
          处理字符串 0cdf5b5d0ce1_01.jpg  => ('0cdf5b5d0ce1_01', '.jpg')
        '''
        self.ids = [splitext(file)[0] for file in listdir(images_dir)]
        # 进行数据增强处理
        if(augment):
            self.ids+=[splitext(file)[0] for file in listdir((aug_dir))]



    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            img_ndarray = img_ndarray / 255

        return img_ndarray

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))
        '''在train_imgs中未找到，在aug_imgs中查找'''
        if(len(img_file)==0):
            img_file=list(self.aug_dir.glob(name+'.*'))

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'

        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)


        # print(Y.shape,torch.as_tensor(mask).shape)

        # return {
        #     'image': torch.as_tensor(img.copy()).float().contiguous(),
        #     'mask': torch.as_tensor(mask.copy()).long().contiguous()
        # }
        return {
            'image': torch.as_tensor(img.copy()).contiguous(),
            'mask': torch.as_tensor(mask.copy()).contiguous()
        }


if  __name__=="__main__":
    train_img = Path('../data/train_imgs/')
    test_img = Path('../data/test_imgs/')
    aug_img = Path('../data/aug_imgs/')
    dir_mask = Path('../data/masks/')
    img_scale=1.0

    # train_set=CarvanaDataset(train_img,dir_mask,img_scale,augment=False,aug_dir=aug_img)
    # val_set=CarvanaDataset(test_img,dir_mask,img_scale)
    #
    # loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    # train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    # val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    img=Image.open("1_mask.gif")
    img_array=np.asarray(img)
    img_tensor=torch.as_tensor(img_array)
    print(img_tensor.shape)
    Y=get_edge(img_tensor,save_mask=True)
    print(Y.shape)
