import argparse
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as F
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import math

from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode, transforms
from PIL import Image
from pytorch_wavelets import DWTForward
from tqdm import tqdm
from omegaconf import OmegaConf

from models.vae import AutoencoderKL
from util.crop import center_crop_arr


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f).convert('RGB')
    return img


class ImageDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self.train_aug = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.resos)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def train_dataloader(self):
        train_set = DatasetFolder(
            root=os.path.join(self.args.data_path, 'train'),
            loader=pil_loader,
            extensions=IMG_EXTENSIONS,
            transform=self.train_aug,
        )
        return DataLoader(
            dataset=train_set,
            batch_size=self.args.batch_size, num_workers=4,
            shuffle=False, drop_last=False,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('AE Analysis', add_help=False)
    parser.add_argument('--batch_size', default=250, type=int)
    parser.add_argument('--data_path', default='/BS/var/nobackup/imagenet-1k/', type=str)
    parser.add_argument('--resos', default=256, type=int)
    parser.add_argument('--model_type', default='marvae', type=str)
    args = parser.parse_args()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # data loading
    image_data_module = ImageDataModule(args)
    image_train_loader = image_data_module.train_dataloader()

    # build the model
    if args.model_type == 'marvae':
        vae = AutoencoderKL(
            embed_dim=16, 
            ddconfig={'z_channels': 16, 'ch_mult': (1, 1, 2, 2, 4)},
            ckpt_path="pretrained_models/vae/kl16.ckpt",
        ).to(device)
    elif args.model_type == 'vavae':
        config = OmegaConf.load('/BS/var/work/LightningDiT/tokenizer/configs/vavae_f16d32.yaml')
        vae = AutoencoderKL(
            embed_dim=32,
            ddconfig=config.model.params.ddconfig,
            ckpt_path='/BS/var/work/LightningDiT/vavae-imagenet256-f16d32-dinov2.pt',
            model_type='vavae',
        ).to(device)
    elif args.model_type == 'vavae-high':
        config = OmegaConf.load('/BS/var/work/LightningDiT/vavae/configs/f16d32_vfdinov2_high.yaml')
        vae = AutoencoderKL(
            embed_dim=32,
            ddconfig=config.model.params.ddconfig,
            ckpt_path='/BS/var/work/LightningDiT/vavae/logs/f16d32_vfdinov2_high/checkpoints/last.ckpt',
            model_type='vavae',
        ).to(device)
    elif args.model_type == 'vavae-low':
        config = OmegaConf.load('/BS/var/work/LightningDiT/vavae/configs/f16d32_vfdinov2_low.yaml')
        vae = AutoencoderKL(
            embed_dim=32, 
            ddconfig=config.model.params.ddconfig,
            ckpt_path='/BS/var/work/LightningDiT/vavae/logs/f16d32_vfdinov2_low/checkpoints/last.ckpt',
            model_type='vavae',
        ).to(device)
    vae.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    print(f'prepare finished.')

    dwt = DWTForward(J=1, wave='haar', mode='zero').to(device)
    
    # Welford's online algorithm
    # https://stackoverflow.com/questions/56402955/whats-the-formula-for-welfords-algorithm-for-variance-std-with-batch-updates
    count = 0 
    mean = 0.0
    M2 = 0.0
    
    # scale_factor = 1 / torch.std(vae.encode(x).sample())
    with torch.no_grad():
        for imgs, labels in tqdm(image_train_loader, desc='Processing images', leave=True):
            x = imgs.to(device)
            if args.model_type == 'vavae-high':
                _, hs = dwt(x)
                x = hs[0] / 2 # normalize
                x = x.view(-1, 9, 128, 128)
            elif args.model_type == 'vavae-low':
                ll1, _ = dwt(x)
                x = ll1 / 2 # normalize
                x = torch.nn.functional.interpolate(x, size=256, mode='bicubic', align_corners=False)

            z = vae.encode(x).sample()
            
            count_new = count + z.numel()
            # delta = x - mean
            delta = z - mean
            mean_new = mean + delta.sum().item() / count_new
            # delta2 = x - new_mean
            delta2 = z - mean_new
            M2 += (delta * delta2).sum().item()

            count = count_new
            mean = mean_new
            
    variance = M2 / (count - 1)
    std = math.sqrt(variance)
    print(f'std: {std}, scale_factor: {1 / std}')
