import argparse
import os
from mxx.ReID.utils.path import load_cfg
from PIL import Image
from tqdm import tqdm
import numpy as np
from mxx.eval.FID import calculate_fid_score
import torch

def add_img(imgs_list, path_img, size_tgt=(128, 256)):
    try:
        img = Image.open(path_img).convert('RGB')
        img = img.resize(size_tgt)
        img = np.array(img)
        imgs_list.append(img) 
    except Exception as e:
        print(f"无法加载图像 {path_img}: {e}")

def eval_kid(path_cfg, dir_sub):
    imgs_tgt_list = []
    imgs_gen_list = []
    cfg = load_cfg(path_cfg)
    dir_ref = cfg["dir"]["ref"]
    for dir in tqdm(os.listdir(cfg["dir"]["ref"])):
        dir = os.path.join(dir_ref, dir)
        if os.path.isdir(dir):
            path_img_tgt = os.path.join(dir, "tgt", "img_0.jpg")
            path_img_gen = os.path.join(dir, dir_sub, "img_0.jpg")
            if os.path.exists(path_img_tgt) and os.path.exists(path_img_gen):
                add_img(imgs_tgt_list, path_img_tgt)
                add_img(imgs_gen_list, path_img_gen)
    
    imgs_tgt = torch.from_numpy(np.array(imgs_tgt_list)).float()  # (N, H, W, C)
    imgs_gen = torch.from_numpy(np.array(imgs_gen_list)).float()
    imgs_tgt = imgs_tgt.permute(0, 3, 1, 2)  # 转换为 (N, C, H, W)
    imgs_gen = imgs_gen.permute(0, 3, 1, 2)

    from torch.hub import load_state_dict_from_url
    import torch_fidelity
    from torchmetrics.image.kid import KernelInceptionDistance
    URL_INCEPTION_V3 = "https://huggingface.co/toshas/torch-fidelity/resolve/main/weights-inception-2015-12-05-6726825d.pth"
    torch_fidelity.feature_extractor_inceptionv3.URL_INCEPTION_V3 = URL_INCEPTION_V3
    kid = KernelInceptionDistance(subset_size=10, normalize=True)  # 关键修改：启用normalize
    kid.update(imgs_tgt, real=True)
    kid.update(imgs_gen, real=False)
    kid_mean, kid_std = kid.compute()
    print(f"KID: {kid_mean:.4f} ± {kid_std:.4f}")

def eval_fid(path_cfg, dir_sub):
    imgs_tgt_list = []
    imgs_gen_list = []
    cfg = load_cfg(path_cfg)
    dir_ref = cfg["dir"]["ref"]
    for dir in tqdm(os.listdir(cfg["dir"]["ref"])):
        dir = os.path.join(dir_ref, dir)
        if os.path.isdir(dir):
            path_img_tgt = os.path.join(dir, "tgt", "img_0.jpg")
            path_img_gen = os.path.join(dir, dir_sub, "img_0.jpg")
            if os.path.exists(path_img_tgt) and os.path.exists(path_img_gen):
                add_img(imgs_tgt_list, path_img_tgt)
                add_img(imgs_gen_list, path_img_gen)
    imgs_gen = np.array(imgs_gen_list)
    imgs_tgt = np.array(imgs_tgt_list)
    from torchmetrics.image.kid import KernelInceptionDistance
    kid = KernelInceptionDistance(subset_size=10)  # 子集大小可调
    kid.update(imgs_tgt, real=True)
    kid.update(imgs_gen, real=False)
    print(kid.compute())
    exit()
    fid_score = calculate_fid_score(imgs_gen, imgs_tgt, "cuda")
    print(fid_score)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_cfg", type=str, default="/machangxiao/code/MIP-ReID/configs/datasets/MARS-v160809/cfg_cache_mars_train.yaml")
    parser.add_argument("--dir_sub", type=str)
    args = parser.parse_args()
    path_cfg = args.path_cfg
    dir_sub = args.dir_sub
    eval_kid(path_cfg=path_cfg, dir_sub=dir_sub)