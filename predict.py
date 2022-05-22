import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

from utils.data_loading import CarvanaDataset
from unet import UNet, UNet1, UNet2, UNet3
from utils.utils import plot_img_and_mask
from os import listdir
from os.path import  splitext
from PIL import  Image
from torchvision import transforms

def dataAug(img_dir,aug_dir,aug_num=5):
    for filename in listdir(img_dir):
        img_path=img_dir+filename
        image=Image.open(img_path)
        aug_op=transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.5)
        for i in range(aug_num):
            aug_img=aug_op(image)
            aug_img.save(aug_dir+"/"+str(i)+filename)


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(CarvanaDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    # parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    # parser.add_argument('--output', '-o', metavar='INPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


if __name__ == '__main__':

    img_dir = "predict_data/img/"
    img_out="predict_data/out_img/"
    aug_dir = "predict_data/aug/"
    aug_out = "predict_data/out_aug/"

    # dataAug(img_dir, aug_dir)

    args = get_args()
    # in_files = args.input
    # out_files = get_output_filenames(args)

    net = UNet(n_channels=3, n_classes=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info('Model loaded!')

    img_ids = [filename for filename in listdir(img_dir)]
    aug_ids=[filename for filename in listdir(aug_dir)]
    for i, filename in enumerate(img_ids):
        logging.info(f'\nPredicting image {filename} ...')
        img = Image.open(img_dir+filename)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)
        result = mask_to_image(mask)
        result.save(img_out+"out_"+filename)

    for i, filename in enumerate(aug_ids):
        logging.info(f'\nPredicting image {filename} ...')
        img = Image.open(aug_dir + filename)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)
        result = mask_to_image(mask)
        result.save(aug_out + "out_" + filename)
        # logging.info(f'Mask saved to {out_filename}')

        # if not args.no_save:
        #     out_filename = out_files[i]
        #     result = mask_to_image(mask)
        #     result.save(out_filename)
        #     logging.info(f'Mask saved to {out_filename}')
        #
        # if args.viz:
        #     logging.info(f'Visualizing results for image {filename}, close to continue...')
        #     plot_img_and_mask(img, mask)
