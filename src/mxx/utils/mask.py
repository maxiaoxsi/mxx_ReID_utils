from PIL import Image
import numpy as np
import os
import errno

def make_mask_img(args):
    (
        path_manikin, 
        path_mask,
    ) = args
    img = Image.open(path_manikin).convert('L')
    arr = np.array(img)
    h, w = arr.shape
    step_h = h // 8
    step_w = w // 8
    for i in range(0, h, step_h):
        for j in range(0, w, step_w):
            block = arr[i:i+step_h, j:j+step_w]
            if np.any(block != 0):
                arr[i:i+step_h, j:j+step_w] = 1
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

def make_back_and_fore_img(path_mask, path_reid, is_save=False, path_back = '', path_fore = ''):
    img_reid = Image.open(path_reid)
    img_mask = Image.open(path_mask)
    arr_mask = np.array(img_mask)
    h, w = arr_mask.shape
    img_reid = img_reid.convert("RGB")
    # Create foreground and background images (RGB, no transparency)
    img_fore = Image.new("RGB", (w, h), (0, 0, 0))  # Black background
    img_back = Image.new("RGB", (w, h), (0, 0, 0))  # Black background
    
    # Get pixel data for efficient access
    pixel_reid = img_reid.load()
    pixel_fore = img_fore.load()
    pixel_back = img_back.load()
    
    for i in range(h):
        for j in range(w):
            if arr_mask[i, j] > 0:  # Foreground pixel
                pixel_fore[j, i] = pixel_reid[j, i]
            else:  # Background pixel
                pixel_back[j, i] = pixel_reid[j, i]
    if is_save:
        img_fore.save(path_fore)
        img_back.save(path_back)
    return img_fore, img_back
    