from PIL import Image
import numpy as np
import random
import os
import errno
from ..ReID.utils.path import get_dir_sub, get_path, get_basename

def make_mask(path_reid, path_manikin, rate_mask_aug):
    img_reid = Image.open(path_reid)
    if not path_manikin or not os.path.exists(path_manikin):
        return Image.new("L", img_reid.size, color=0), img_reid, Image.new("RGB", img_reid.size, color="black")
    img = Image.open(path_manikin).convert('L')
    arr = np.array(img)
    h, w = arr.shape
    step_h = h // 8
    step_w = w // 8
    cache = []
    for i in range(0, h, step_h):
        for j in range(0, w, step_w):
            end_i = min(i + step_h, h)
            end_j = min(j + step_w, w)
            block = arr[i:end_i, j:end_j]
            start_i = max(0, i - step_h // 8)
            end_i = min(i + step_h + step_h // 8, h)
            start_j = max(0, j - step_w // 8)
            end_j = min(j + step_w + step_w // 8, w)
            if np.any(block > 10):
                cache.append((start_i, end_i, start_j, end_j))
            else:
                if random.random() < rate_mask_aug:
                    cache.append((start_i, end_i, start_j, end_j))
                else:
                    arr[start_i:end_i, start_j:end_j] = 0
    for (start_i, end_i, start_j, end_j) in cache:
        arr[start_i:end_i, start_j:end_j] = 1
        
    img_mask = Image.fromarray((arr > 0).astype(np.uint8) * 255)
    img_fore = Image.new("RGB", (w, h), (0, 0, 0))  # Black background
    img_back = Image.new("RGB", (w, h), (0, 0, 0))  # Black background
    pixel_reid = img_reid.load()
    pixel_fore = img_fore.load()
    pixel_back = img_back.load()
    for i in range(h):
        for j in range(w):
            if arr[i, j] > 0:  # Foreground pixel
                pixel_fore[j, i] = pixel_reid[j, i]
            else:  # Background pixel
                pixel_back[j, i] = pixel_reid[j, i]
    return img_mask, img_fore, img_back


def make_mask_img(args):
    (cfg, root, file, logger) = args
    if not file.endswith(('.jpg', '.png')):
        return
    dir_sub = get_dir_sub(root, cfg)
    basename, ext = get_basename(file)
    path_manikin = get_path(cfg, dir_sub, basename, ext, "manikin")
    if not os.path.exists(path_manikin):
        return
    path_mask = get_path(cfg, dir_sub, basename, ext, "mask")
    img = Image.open(path_manikin).convert('L')
    arr = np.array(img)
    h, w = arr.shape
    step_h = h // 10
    step_w = w // 8
    for i in range(0, h, step_h):
        for j in range(0, w, step_w):
            end_i = min(i + step_h, h)
            end_j = min(j + step_w, w)
            block = arr[i:end_i, j:end_j]
            if np.any(block > 10):
                arr[i:i+step_h, j:j+step_w] = 1
            else:
                arr[i:i+step_h, j:j+step_w] = 0
    img_mask = Image.fromarray((arr > 0).astype(np.uint8) * 255)
    dir_mask = os.path.dirname(path_mask)
    try:
        os.makedirs(dir_mask)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    img_mask.save(path_mask)
    # print(f"mask img saved to {path_mask}")
    return arr


    