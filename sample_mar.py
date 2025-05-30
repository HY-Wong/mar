import os
import torch
import torchvision
import torch_fidelity
import numpy as np

from PIL import Image
from models import mar
from models.vae import AutoencoderKL


device = "cuda" if torch.cuda.is_available() else "cpu"

model_type = "mar_base" #@param ["mar_base", "mar_large", "mar_huge"]
num_sampling_steps_diffloss = 100 #@param {type:"slider", min:1, max:1000, step:1}

if model_type == "mar_base":
    diffloss_d = 6
    diffloss_w = 1024
elif model_type == "mar_large":
    diffloss_d = 8
    diffloss_w = 1280
elif model_type == "mar_huge":
    diffloss_d = 12
    diffloss_w = 1536
else:
    raise NotImplementedError

model = mar.__dict__[model_type](
    buffer_size=64,
    diffloss_d=diffloss_d,
    diffloss_w=diffloss_w,
    num_sampling_steps=str(num_sampling_steps_diffloss),
).to(device)

state_dict = torch.load("pretrained_models/mar/{}/checkpoint-last.pth".format(model_type))["model_ema"]
model.load_state_dict(state_dict)
model.eval() # important!
vae = AutoencoderKL(
    embed_dim=16, 
    ddconfig={'z_channels': 16, 'ch_mult': (1, 1, 2, 2, 4)},
    ckpt_path="pretrained_models/vae/kl16.ckpt",
).to(device)
vae.eval()

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
    sampled_images = vae.decode(sampled_tokens / 0.2325)

# Save and display images:
sampled_images = (sampled_images + 1) / 2
chw = torchvision.utils.make_grid(sampled_images, nrow=8, padding=0, pad_value=1.0)
chw = np.round(np.clip(chw.permute(1, 2, 0).mul_(255).cpu().numpy(), 0, 255))
chw = Image.fromarray(chw.astype(np.uint8))
chw.save('sample_mar.png')