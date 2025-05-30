import os
import torch
import torchvision
import torch_fidelity
import numpy as np

from PIL import Image
from models import mar
from models.vae import AutoencoderKL
from omegaconf import OmegaConf
from pytorch_wavelets import DWTInverse


device = "cuda" if torch.cuda.is_available() else "cpu"

num_sampling_steps_diffloss = 100 #@param {type:"slider", min:1, max:1000, step:1}
model = mar.__dict__['mar_large'](
    img_size=256,
    vae_stride=16,
    patch_size=1,
    vae_embed_dim=64,
    mask_ratio_min=0.7,
    label_drop_prob=0.1,
    class_num=1000,
    attn_dropout=0.1,
    proj_dropout=0.1,
    buffer_size=64,
    diffloss_d=3,
    diffloss_w=1024,
    num_sampling_steps=str(num_sampling_steps_diffloss),
    diffusion_batch_mul=4,
).to(device)

state_dict = torch.load('/BS/var/work/mar/output_dir/checkpoint-last.pth', weights_only=False)["model_ema"]
model.load_state_dict(state_dict)
model.eval() # important!
config_high = OmegaConf.load('vavae/configs/f16d32_vfdinov2_high.yaml')
vae_high = AutoencoderKL(
    embed_dim=config_high.model.params.embed_dim,
    ddconfig=config_high.model.params.ddconfig,
    ckpt_path='/BS/var/work/LightningDiT/vavae/logs/f16d32_vfdinov2_high/checkpoints/last.ckpt',
    model_type='vavae',
).cuda().eval()
for param in vae_high.parameters():
    param.requires_grad = False

config_low = OmegaConf.load('vavae/configs/f16d32_vfdinov2_low.yaml')
vae_low = AutoencoderKL(
    embed_dim=config_low.model.params.embed_dim, 
    ddconfig=config_low.model.params.ddconfig,
    ckpt_path='/BS/var/work/LightningDiT/vavae/logs/f16d32_vfdinov2_low/checkpoints/last.ckpt',
    model_type='vavae',
).cuda().eval()
for param in vae_low.parameters():
    param.requires_grad = False

idwt = DWTInverse('haar', mode='zero').to(device)

# Set user inputs:
seed = 0 #@param {type:"number"}
torch.manual_seed(seed)
np.random.seed(seed)
num_ar_steps = 64 #@param {type:"slider", min:1, max:256, step:1}
cfg_scale = 4 #@param {type:"slider", min:1, max:10, step:0.1}
cfg_schedule = "constant" #@param ["linear", "constant"]
temperature = 1.0 #@param {type:"slider", min:0.9, max:1.1, step:0.01}
class_labels = 0, 1, 2, 3, 4, 5, 6, 7 #@param {type:"raw"}

with torch.cuda.amp.autocast():
    sampled_tokens = model.sample_tokens(
        bsz=len(class_labels), num_iter=num_ar_steps,
        cfg=cfg_scale, cfg_schedule=cfg_schedule,
        labels=torch.Tensor(class_labels).long().cuda(),
        temperature=temperature, progress=True,
    )
    sampled_tokens_high, sampled_tokens_low = torch.chunk(sampled_tokens, 2, dim=1)
                    
    # high frequency components
    sampled_high = vae_high.decode(sampled_tokens_high / 0.9072)
    sampled_high = sampled_high.view(-1, 3, 3, 128, 128)
    sampled_high = sampled_high * 2 # denormalize

    # low frequency components
    sampled_low = vae_low.decode(sampled_tokens_low / 0.2917)
    sampled_low = torch.nn.functional.interpolate(sampled_low, size=128, mode='bicubic', align_corners=False)
    sampled_low = sampled_low * 2 # denormalize
    
    # reconstruct images from wavelet components
    sampled_images = idwt((sampled_low, [sampled_high]))

# Save and display images:
sampled_images = (sampled_images + 1) / 2
chw = torchvision.utils.make_grid(sampled_images, nrow=8, padding=0, pad_value=1.0)
chw = np.round(np.clip(chw.permute(1, 2, 0).mul_(255).cpu().numpy(), 0, 255))
chw = Image.fromarray(chw.astype(np.uint8))
chw.save('sample_mar-va-vae-high-low-f16c32.png')