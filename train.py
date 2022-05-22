import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torchvision import transforms
import torchvision.transforms.functional as TF

from utils.data_loading import CarvanaDataset
from utils.dice_score import dice_loss
from utils.edge_score import edge_loss
from evaluate import evaluate
from unet import UNet, UNet1, UNet2, UNet3
from dataProcess.data_augment import Augment, del_file

import pandas as pd

train_img = Path('./data/train_imgs/')
test_img = Path('./data/test_imgs/')
aug_img = Path('./data/aug_imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')


def data_loading(isAugment, img_scale, batch_size):
    if isAugment:
        del_file(dir="./data/aug_imgs/")
        for i in aug_index:
            augment = Augment(train_img, "./data/aug_imgs/", aug_op=aug_fun[i], aug_suffix=aug_name[i],
                              mask_dir="./data/masks/")
            augment.process()

    train_set = CarvanaDataset(train_img, dir_mask, img_scale, augment=isAugment, aug_dir=aug_img)
    n_train = train_set.__len__()
    logging.info(f'Creating train_set with {n_train} examples')

    val_set = CarvanaDataset(test_img, dir_mask, img_scale)
    n_val = val_set.__len__()
    logging.info(f'Creating val_set with {n_val} examples')

    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    return n_train, train_loader, n_val, val_loader


def train_net(net,
              model_name,
              device,
              aug_index,
              aug_fun,
              aug_name,
              isAugment=False,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 1e-5,
              save_checkpoint: bool = True,

              edgeloss: bool = False,
              edge_weight: int = 50,
              outline_weight: int = 0,
              img_scale: float = 1,

              amp: bool = False):
    n_train, train_loader, n_val, val_loader = data_loading(isAugment, img_scale, batch_size)

    experiment = wandb.init(project='Few-shot-semantic-segmentation', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, img_scale=img_scale,
             amp=amp, isAugment=isAugment))

    if not isAugment :
        aug_index = [False] * len(aug_index)

    data = {}
    data['model'] = [model_name]
    for i in range(len(aug_name)):
        if i in aug_index and isAugment:
            data[aug_name[i]] = True
        else:
            data[aug_name[i]] = False
    data['edgeLoss'] = [edgeloss]
    data['edge_weight'] = [edge_weight]
    data['outline_weight'] = [outline_weight]
    data['edgeLoss'] = [edgeloss]
    df = pd.DataFrame(data)

    tbl = wandb.Table(data=df)
    experiment.log({'augment_table': tbl})

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        augIndex:        {aug_index}
        augList:         {aug_name}
        Images scaling:  {img_scale}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()

    global_step = 0
    imgs = 0
    rounds=0
    # 5. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        cur_step=0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']
                assert images.shape[1] == net.n_channels, f'channels mismatch'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    # true_masks:[n,h,w]   masks_pred:[n,nclasses,h,w]
                    if edgeloss:
                        loss = criterion(masks_pred, true_masks) \
                               + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                           F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                           multiclass=True) \
                               + edge_loss(input=torch.softmax(masks_pred, dim=1), target=true_masks
                                           , edge_weight=edge_weight, outline_weight=outline_weight, multiclass=True)
                        '''
                        edge_loss(mask_predit=torch.softmax(masks_pred, dim=1).argmax(dim=1)[0],
                                            mask_true=true_masks[0], device=device,edge_weight=edge_weight,outline_weight=outline_weight)
                        '''
                    else:
                        loss = criterion(masks_pred, true_masks) \
                               + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                           F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                           multiclass=True)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)  # Internally invokes unscale_(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                cur_step+=1
                imgs+=images.shape[0]           #当前batch训练的图片数
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'imgs': imgs,
                    'epoch':epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round

                division_step = (n_train // (10 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        rounds+=1
                        # train_score=0
                        train_score = evaluate(net, train_loader, device)
                        val_score = evaluate(net, val_loader, device)

                        scheduler.step(val_score)

                        logging.info('Train loss:{}  Train dice score:{} Validation Dice score:{}'.format(epoch_loss / cur_step,train_score,val_score))
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'train Loss':epoch_loss / cur_step,
                            'train Dice':train_score,
                            'validation Dice': val_score,
                            'images': wandb.Image(images[0].cpu()),
                            'masks': {
                                'true': wandb.Image(true_masks[0].float().cpu()),
                                'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                            },
                            'imgs': global_step,
                            'rounds':rounds
                        })
                        if rounds%10==0 and rounds>0:
                            experiment.log({
                                'train Loss': epoch_loss / cur_step,
                                'train Dice': train_score,
                                'validation Dice': val_score,
                                'epoch':epoch+1
                            })
        # train_score=evaluate(net,train_loader,device)
        # val_score = evaluate(net, val_loader, device)
        # logging.info('train loss: {}'.format(epoch_loss/imgs))
        # logging.info('Train Dice score: {}'.format(train_score))
        # logging.info('Validation Dice score: {}'.format(val_score))
        # scheduler.step(val_score)
        # experiment.log({
        #     'learning rate': optimizer.param_groups[0]['lr'],
        #     'train loss':epoch_loss/imgs,
        #     'train dice':train_score,
        #     'validation Dice': val_score,
        #     'epoch':epoch+1
        # })
        # logging.info("epoch_val_mean:{}".format(epoch_val / 10))
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            aug_suffix = ""
            if isAugment:
                aug_suffix += "_aug"
                for  value in aug_index:
                    aug_suffix += aug_name[value]
                    # if value:
                    #     aug_suffix += aug_name[i]

            if edgeloss:
                torch.save(net.state_dict(),
                           str(dir_checkpoint / '{}_epoch{}{}_edge{}_outline{}.pth'.format(model_name, epoch + 1,
                                                                                           aug_suffix, edge_weight,
                                                                                           outline_weight)))
            else:
                torch.save(net.state_dict(),
                           str(dir_checkpoint / '{}_epoch{}{}.pth'.format(model_name, epoch + 1, aug_suffix)))
            logging.info(f'Checkpoint {epoch + 1} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--model', '-m', metavar="model", type=str, default="unet", help="The model that use")
    parser.add_argument('--epochs', '-e', metavar='epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--augment', '-aug', metavar='augment', type=bool, default=False,
                        help="apply data augmentation")

    # parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5, help='Learning rate',dest='lr')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model_name = args.model
    logging.info(f'Using model {args.model}')

    if model_name == "unet":
        net = UNet(n_channels=3, n_classes=2)
    elif model_name == 'unet1':
        net = UNet1(n_channels=3, n_classes=2)
    elif model_name == 'unet2':
        net = UNet2(n_channels=3, n_classes=2)
    elif model_name == 'unet3':
        net = UNet3(n_channels=3, n_classes=2)

    logging.info(f'Network: {net.n_channels} input channels {net.n_classes} output channels (classes)')

    # if args.load:
    #     net.load_state_dict(torch.load(args.load, map_location=device))
    #     logging.info(f'Model loaded from {args.load}')

    # 要进行的增强操作
    aug_index = [1,2,3,6,7]
    aug_fun = [TF.invert,
               TF.hflip,
               TF.rotate,
               TF.affine,  # 3 scale=1.5
               TF.affine,  # 4 translateX=200
               transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
               transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.5),
               transforms.ColorJitter(brightness=0, contrast=0.5, saturation=0, hue=0)]
    aug_name = ["_invert", "_hflip", "_rotate", "_affineScale", "_translateX", "_GaussianBlur", "_ColorJitter_hue0.5",
                "_ColorJitter_contrast0.5"]
    '''
        是否需要对分割图变换
    '''
    mask_trans = ["_hflip", "_rotate"]

    edgeloss = False
    edge_weight = 1
    outline_weight = 2

    net.to(device=device)
    try:
        train_net(net=net,
                  isAugment=args.augment,
                  aug_index=aug_index,
                  aug_fun=aug_fun,
                  aug_name=aug_name,
                  epochs=args.epochs,
                  batch_size=1,
                  learning_rate=1e-5,
                  device=device,
                  img_scale=0.5,
                  edgeloss=edgeloss,
                  edge_weight=edge_weight,
                  outline_weight=outline_weight,
                  model_name=model_name,
                  amp=args.amp)
    except KeyboardInterrupt:
        sys.exit(0)
