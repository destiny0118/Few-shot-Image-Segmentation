from pathlib import Path
import os
from os import listdir
from os.path import splitext
from PIL import Image
from torchvision import transforms
import logging
import torchvision.transforms.functional as TF

class Augment():
    def __init__(self,img_dir,aug_dir,aug_op,aug_suffix,mask_dir,mask_suffix='_mask',mask_aug=False):
        # 封装图片路径，获取文件
        self.img_dir=Path(img_dir)
        self.aug_dir=aug_dir
        self.aug_op=aug_op
        self.aug_suffix=aug_suffix
        self.mask_dir=Path(mask_dir)
        self.mask_suffix=mask_suffix
        self.mask_aug=mask_aug

        self.ids=[splitext(filename)[0] for filename in listdir(img_dir)]
        # self.ids += [splitext(filename)[0] for filename in listdir(img_dir)]
        # print(len(self.ids))

    '''
        图片路径，打开的图片，增强操作
        
    '''
    def apply(self,index,image,mask):
        aug_img=self.aug_op(image)
        if self.mask_aug:
            aug_mask=self.aug_op(mask)
        else:
            aug_mask=mask
            # print(self.aug_suffix+"*****")
        aug_img.save(self.aug_dir+index+self.aug_suffix+".jpg")
        aug_mask.save(str(self.mask_dir)+"/"+index+self.aug_suffix+"_mask"+".gif")

    def process(self):
        for index in self.ids:
            img_path=list(self.img_dir.glob(index+".*"))
            mask_path=list(self.mask_dir.glob(index+self.mask_suffix+".*"))

            image=Image.open(img_path[0])
            mask=Image.open(mask_path[0])
            self.apply(index,image,mask)
        logging.info(f'Creating {len(self.ids)}  {self.aug_suffix} augment images')

def del_file(dir):
    for file in listdir(dir):
        file_path=dir+file
        if(os.path.isfile(file_path)):
            os.remove(file_path)
            # else:
            #     self.del_file(file_path)


if __name__=="__main__":
    aug_list=[transforms.RandomHorizontalFlip(1),
              transforms.ColorJitter(brightness=0,contrast=0,saturation=0,hue=0.5)]
    suffix_list=["_HF","_CJ_HUE"]
    del_file("../data/aug_imgs/")
    # for i in range(len(suffix_list)):
    #     augment=Augment(img_dir="../data/train_imgs/",aug_dir="../data/aug_imgs/",
    #                     aug_op=aug_list[i],aug_suffix=suffix_list[i],
    #                     mask_dir="../data/masks/")
    #     # augment.process()
    #     augment.del_file("../data/aug_imgs/")