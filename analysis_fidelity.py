import argparse
import os
import torch
import torchvision
import torchvision.transforms.functional as F
import numpy as np
import torch_fidelity
import cv2

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from PIL import Image
from tqdm import tqdm

from models.vae import AutoencoderKL


if __name__ == '__main__':
    parser = argparse.ArgumentParser('VAE Analysis', add_help=False)
    parser.add_argument('--batch_size', default=200, type=int)
    parser.add_argument('--data_path', default='/path/to/imagenet-1k/', type=str)
    parser.add_argument('--resos', default=256, type=int)
    args = parser.parse_args()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # data loading
    transform = transforms.Compose([
        transforms.Resize(args.resos),
        transforms.CenterCrop((args.resos, args.resos)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = ImageFolder(root=os.path.join(args.data_path, 'val'), transform=transform)
    image_val_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # build the model
    vae = AutoencoderKL(
        embed_dim=16, 
        ddconfig={'z_channels': 16, 'ch_mult': (1, 1, 2, 2, 4)},
        ckpt_path="pretrained_models/vae/kl16.ckpt",
    ).to(device)
    vae.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    print(f'prepare finished.')

    ref_dir = '/path/to/recon/ground_truth' # reference images
    save_dir = '/path/to/recon/kl-vae-f16c16-mar' # generated images
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # save reference images if needed
    ref_png_files = [f for f in os.listdir(ref_dir) if f.endswith('.png')]
    save_ref = bool(len(ref_png_files) < 50000)
    
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(tqdm(image_val_loader, desc='Processing images', leave=True)):
            imgs = imgs.to(device)
            posterior = vae.encode(imgs)
            z = posterior.sample()
            rec_imgs = vae.decode(z)
            
            if save_ref:
                imgs = torch.clamp((imgs + 1) / 2, min=0, max=1)
                for b_id in range(imgs.size(0)):
                    img_id = i * imgs.size(0) + b_id
                    img = np.round(imgs[b_id].cpu().numpy().transpose([1, 2, 0]) * 255)
                    img = img.astype(np.uint8)[:, :, ::-1]
                    cv2.imwrite(os.path.join(ref_dir, f'{img_id:05}.png'), img)

            rec_imgs = torch.clamp((rec_imgs + 1) / 2, min=0, max=1)
            for b_id in range(rec_imgs.size(0)):
                img_id = i * rec_imgs.size(0) + b_id
                img = np.round(rec_imgs[b_id].cpu().numpy().transpose([1, 2, 0]) * 255)
                img = img.astype(np.uint8)[:, :, ::-1]
                cv2.imwrite(os.path.join(save_dir, f'{img_id:05}.png'), img)

    metrics_dict = torch_fidelity.calculate_metrics(
        input1=save_dir,
        input2=ref_dir,
        cuda=True,
        isc=True,
        fid=True,
        kid=False,
        prc=False,
        verbose=True,
    )
    fid = metrics_dict['frechet_inception_distance']
    inception_score = metrics_dict['inception_score_mean']
    print(f'{fid=}, {inception_score=}')
