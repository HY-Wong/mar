import argparse
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as F
import numpy as np
import lpips
import clip

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from PIL import Image
from pytorch_wavelets import DWTForward
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm

from models.vae import AutoencoderKL


if __name__ == '__main__':
    parser = argparse.ArgumentParser('VAE Analysis', add_help=False)
    parser.add_argument('--batch_size', default=10, type=int)
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
    dataset = ImageFolder(root=os.path.join(args.data_path, 'hard'), transform=transform)
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
    
    # evaluate the reconstruction loss [-1, 1] range 
    total_loss = {key: 0.0 for key in ['reconstruction', 'low_frequency', 'high_frequency', 'perceptual', 'clip_semantic', 'sam_semantic']}
    total_images = 0
    visualize_imgs = {key: [] for key in ['rec_imgs', 'rec_ll1', 'rec_lh1', 'rec_hl1', 'rec_hh1']}
    
    # for evaluation metrics
    dwt = DWTForward(J=1, wave='haar', mode='zero').to(device)
    l2_loss = nn.MSELoss(reduction='mean')
    
    lpips_loss = lpips.LPIPS(net='vgg').to(device).eval()
    
    clip, preprocess_clip = clip.load('ViT-B/32', device=device)
    clip.eval()
    
    sam = sam_model_registry['vit_b'](checkpoint='/path/to/sam_vit_b_01ec64.pth')
    sam = sam.to(device).eval()
    sam_predictor = SamPredictor(sam)

    with torch.no_grad():
        for imgs, labels in tqdm(image_val_loader, desc='Processing images', leave=True):
            imgs = imgs.to(device)
            posterior = vae.encode(imgs)
            z = posterior.sample()
            rec_imgs = vae.decode(z)
            
            # first level DWT
            ll1, hs = dwt(imgs)
            lh1, hl1, hh1 = hs[0][:, 0], hs[0][:, 1], hs[0][:, 2]
            rec_ll1, rec_hs = dwt(rec_imgs)
            rec_lh1, rec_hl1, rec_hh1 = rec_hs[0][:, 0], rec_hs[0][:, 1], rec_hs[0][:, 2]
            
            # preprocess for CLIP, which expects input of size (224, 224)
            # more efficient than applying preprocess_clip() sample-by-sample, but introduce slight discrepancies 
            # due to differences between PIL and tensor-based Resize implementations in torchvision
            imgs_clip = torch.clamp((imgs + 1) / 2, min=0, max=1)
            imgs_clip = F.resize(imgs_clip, 224, F.InterpolationMode.BICUBIC)
            imgs_clip = F.normalize(imgs_clip, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            rec_imgs_clip = torch.clamp((rec_imgs + 1) / 2, min=0, max=1)
            rec_imgs_clip = F.resize(rec_imgs_clip, 224, F.InterpolationMode.BICUBIC)
            rec_imgs_clip = F.normalize(rec_imgs_clip, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

            # imgs_clip = torch.clamp((imgs + 1) / 2, min=0, max=1).cpu()
            # imgs_clip = torch.stack([preprocess_clip(torchvision.transforms.ToPILImage()(img)) for img in imgs_clip]).to(device)
            # rec_imgs_clip = torch.clamp((rec_imgs + 1) / 2, min=0, max=1).cpu()
            # rec_imgs_clip = torch.stack([preprocess_clip(torchvision.transforms.ToPILImage()(img)) for img in rec_imgs_clip]).to(device)
            
            features_clip = clip.encode_image(imgs_clip)
            rec_features_clip = clip.encode_image(rec_imgs_clip)

            # preprocess for SAM, which expects input of size (1024, 1024)
            # slightly differs from apply_image(), which uses uint8 NumPy arrays
            imgs_sam = torch.clamp((imgs + 1) / 2, min=0, max=1).mul_(255)
            imgs_sam = sam_predictor.transform.apply_image_torch(imgs_sam)
            rec_imgs_sam = torch.clamp((rec_imgs + 1) / 2, min=0, max=1).mul_(255)
            rec_imgs_sam = sam_predictor.transform.apply_image_torch(rec_imgs_sam)
            
            sam_predictor.set_torch_image(imgs_sam, (256, 256))
            features_sam = sam_predictor.features
            features_sam = features_sam.reshape(imgs.shape[0], -1)
            sam_predictor.set_torch_image(rec_imgs_sam, (256, 256))
            rec_features_sam = sam_predictor.features
            rec_features_sam = rec_features_sam.reshape(imgs.shape[0], -1)

            batch_losses = {
                'reconstruction': l2_loss(rec_imgs, imgs).item(),
                'low_frequency': l2_loss(rec_ll1, ll1).item(),
                'high_frequency': (l2_loss(rec_lh1, lh1).item() + l2_loss(rec_hl1, hl1).item() + l2_loss(rec_hh1, hh1).item()) / 3,
                'perceptual': lpips_loss(rec_imgs, imgs).mean().item(),
                'clip_semantic': 1 - nn.functional.cosine_similarity(features_clip, rec_features_clip).mean().item(),
                'sam_semantic': 1 - nn.functional.cosine_similarity(features_sam, rec_features_sam).mean().item(),
            }

            for key in total_loss:
                total_loss[key] += batch_losses[key] * imgs.shape[0]

            total_images += imgs.shape[0]
            
            # visualize reconstructed images
            rec_ll1, rec_lh1, rec_hl1, rec_hh1 = rec_ll1 / 2, rec_lh1 / 2, rec_hl1 / 2, rec_hh1 / 2
            keys = ['rec_imgs', 'rec_ll1', 'rec_lh1', 'rec_hl1', 'rec_hh1']
            rec_imgs_list = [rec_imgs, rec_ll1, rec_lh1, rec_hl1, rec_hh1]
            for key, rec_imgs in zip(keys, rec_imgs_list):
                visualize_imgs[key].append(rec_imgs)

    save_dir = 'outputs/analysis_figures'
    os.makedirs(save_dir, exist_ok=True)
    
    suffixes = ['', '_ll1', '_lh1', '_hl1', '_hh1']
    for key, suffix in zip(visualize_imgs.keys(), suffixes): 
        rec_imgs = torch.cat(visualize_imgs[key], dim=0)
        rec_imgs = torch.clamp((rec_imgs + 1) / 2, min=0, max=1)
        rec_imgs = torchvision.utils.make_grid(rec_imgs, nrow=4, padding=0)
        rec_imgs = rec_imgs.permute(1, 2, 0).cpu().mul_(255).numpy()
        rec_imgs = Image.fromarray(rec_imgs.astype(np.uint8))
        rec_imgs.save(f'{save_dir}/recon-hard_kl-vae-f16c16-mar{suffix}.png')

    # compute average loss per component
    avg_loss = {key: total_loss[key] / total_images for key in total_loss}
    print(avg_loss)
