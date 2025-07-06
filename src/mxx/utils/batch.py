from math import log
import os
import yaml
from ..annot.annot_base import AnnotBase
from ..log.logger import Logger
from tqdm import tqdm
import numpy as np
from ..qwen_vl.qwen import get_is_backpack

from .mask import make_mask_img

from concurrent.futures import ProcessPoolExecutor
import glob

def check_cfg_dir(dir):
    if not os.path.exists(dir):
        raise Exception(f"dir:{dir} not exists!")

def load_cfg(path_cfg):
    if not os.path.exists(path_cfg):
        print(path_cfg)
        raise Exception("dataset cfg file not found!")
    with open(path_cfg, 'r') as f:
        cfg = yaml.safe_load(f)
    if 'dir' not in cfg:
        raise Exception("dir not in dataset cfg file!")
    check_cfg_dir(cfg['dir']['reid'])
    check_cfg_dir(cfg['dir']['smplx'])
    check_cfg_dir(cfg['dir']['annot'])
    check_cfg_dir(cfg['dir']['mask'])
    return cfg

def load_name(file):
    name_file = file.split('/')[-1]
    suff_file = name_file.split('.')[-1]
    name_file = name_file.split('.')[0]
    return name_file, suff_file


def count_files(path):
    total = 0
    for root, dirs, files in os.walk(path):
        total += len(files)
    return total

def process_batch(
    path_cfg, 
    name_proc, 
    method,
    batch_size,
    max_workers=4,
):
    cfg = load_cfg(path_cfg)
    dir_reid = cfg["dir"]["reid"]
    n_files = count_files(dir_reid)
    logger = Logger('./batch.log')
    data_list = []
    with tqdm(total=n_files, desc=f"Processing {name_proc}") as pbar:
        for root, dirs, files in os.walk(dir_reid):
            for file in files:
                data_list.append((cfg, root, file, logger))
                if len(data_list) == batch_size:
                    with ProcessPoolExecutor(max_workers=max_workers) as executor:
                        list(executor.map(method, data_list))
                    data_list = []
                    pbar.update(batch_size)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(method, data_list))
        pbar.update(len(data_list))



def process_smplx_batch(path_cfg):
    from ..smplx.smplx import render_skeleton
    cfg = load_cfg(path_cfg)
    dir_reid = cfg["dir"]["reid"]
    n_files = count_files(dir_reid)
    data_list = []
    with tqdm(total=n_files, desc="Processing get skeleton") as pbar:
        for root, dirs, files in os.walk(dir_reid):
            for dir in dirs:
                pbar.update(1)
            for file in files:
                if not file.endswith(('.jpg', '.png')):
                    pbar.update(1)
                    continue
                dir_reid = cfg["dir"]["reid"]
                dir_smplx = cfg["dir"]["smplx"]
                dir_sub = root[len(dir_reid) + 1:]
                
                data_list.append((path_reid, path_pred, path_skeleton))
                if len(data_list) == 64:
                    with ProcessPoolExecutor(max_workers=8) as executor:
                        list(executor.map(render_skeleton, data_list))
                    data_list = []
                    pbar.update(64)
        with ProcessPoolExecutor(max_workers=8) as executor:
            list(executor.map(render_skeleton, data_list))
            pbar.update(len(data_list))    

def rename_guidance(cfg, root, dir):
    dir_smplx = cfg["dir"]["smplx"]
    dir_guidance = os.path.join(dir_smplx, dir, 'smplx')
    dir_manikin = os.path.join(dir_smplx, dir, 'pred')
    if os.path.exists(dir_guidance):
        os.rename(dir_guidance, dir_manikin)
    else:
        print("kkkkkk")
    return
        
def process_make_mask(path_cfg):
    cfg = load_cfg(path_cfg)
    dir_reid = cfg["dir"]["reid"]
    n_files = count_files(dir_reid)
    data_list = []
    from .mask import make_mask_img
    with tqdm(total=n_files, desc="Processing make mask") as pbar:
        for root, dirs, files in os.walk(dir_reid):
            for dir in dirs:
                pbar.update(1)
            for file in files:
                if not file.endswith(('.jpg', '.png')):
                    pbar.update(1)
                    continue
                dir_reid = cfg["dir"]["reid"]
                dir_smplx = cfg["dir"]["smplx"]
                dir_mask = cfg["dir"]["mask"]
                dir_sub = root[len(dir_reid) + 1:]
                name_file = file.split('/')[-1]
                suff_file = name_file.split('.')[-1]
                name_file = name_file.split('.')[0]
                path_reid = os.path.join(root, file)
                path_mask = os.path.join(dir_mask, dir_sub, file)
                path_manikin = os.path.join(dir_smplx,
                    dir_sub, 'manikin', file)
                if not os.path.exists(path_manikin):
                    pbar.update(1)
                    continue
                if os.path.exists(path_mask):
                    pbar.update(1)
                    continue
                data_list.append((path_manikin, path_mask))
                if len(data_list) == 128:
                    with ProcessPoolExecutor(max_workers=8) as executor:
                        list(executor.map(make_mask_img, data_list))
                    data_list = []
                    pbar.update(128)
        with ProcessPoolExecutor(max_workers=8) as executor:
            list(executor.map(make_mask_img, data_list))
            pbar.update(len(data_list))  