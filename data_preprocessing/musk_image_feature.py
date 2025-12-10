import torch
from timm.models import create_model
import sys
sys.path.insert(0, "/remote-home/share/lisj/Workspace/SOTA_NAS/encoder/musk/MUSK-main")
from musk import utils, modeling
from PIL import Image
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
import torchvision
import os
import glob
from tqdm import tqdm

device = torch.device("cuda:5")
train_root = "/remote-home/share/lisj/Workspace/SOTA_NAS/datasets/core/patches/train"
out_root   = "/remote-home/share/lisj/Workspace/SOTA_NAS/datasets/core/MUSK-feature"
os.makedirs(out_root, exist_ok=True)

# 加载模型
model_config = "musk_large_patch16_384"
model = create_model(model_config).to(device, dtype=torch.float16).eval()
local_ckpt = "/remote-home/share/lisj/Workspace/SOTA_NAS/encoder/musk/checkpoint/model.safetensors"
utils.load_model_and_may_interpolate(local_ckpt, model, 'model|module', '')
model.eval()

# 先统计一下总共多少张图片
case_dirs = [d for d in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, d))]
case_images = {case: glob.glob(os.path.join(train_root, case, '*.png')) for case in case_dirs}
total_imgs = sum(len(imgs) for imgs in case_images.values())

# 准备整体进度条
pbar = tqdm(total=total_imgs, desc="All images")

img_size = 384
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(img_size, interpolation=3, antialias=True),
    torchvision.transforms.CenterCrop((img_size, img_size)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
])

for case, img_list in case_images.items():
    out_path = os.path.join(out_root, f"{case}.pt")
    # 如果已经处理过，跳过并把该 case 的图片数也加到进度条
    if os.path.exists(out_path):
        pbar.update(len(img_list))
        print(f"-> skip {case}, {len(img_list)} images")
        continue

    features = {}
    for img_path in img_list:
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device, torch.float16)
        with torch.inference_mode():
            emb = model(
                image=img_tensor,
                with_head=False,
                out_norm=True,
                ms_aug=True
            )[0]
        features[os.path.basename(img_path)] = emb.squeeze(0).cpu()
        pbar.update(1)

    torch.save(features, out_path)
    print(f"-> saved {len(features)} embeddings to {out_path}")

pbar.close()
