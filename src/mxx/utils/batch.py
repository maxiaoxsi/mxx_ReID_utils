from math import log
import os
import yaml
from ..annot.annot_base import AnnotBase
from ..log.logger import Logger
from tqdm import tqdm
import numpy as np

from concurrent.futures import ProcessPoolExecutor
import glob

from ..ReID.utils.path import load_cfg

def count_files(path):
    total = 0
    for root, dirs, files in os.walk(path):
        total += len(files)
    return total

def process_reid_batch(
    path_cfg, 
    name_processing, 
    method,
    batch_size,
    max_workers=4,
):
    cfg = load_cfg(path_cfg)
    dir_reid = cfg["dir"]["reid"]
    n_files = count_files(dir_reid)
    logger = Logger('./annot_dataset_reid.log')
    data_list = []
    with tqdm(total=n_files, desc=f"Processing {name_processing}") as pbar:
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

def process_reid_batch_vl(
    path_cfg, 
    keys_text,
    name_processing, 
    method_batch,
    idx_annot,
    batch_size,
):
    cfg = load_cfg(path_cfg)
    dir_reid = cfg["dir"]["reid"]
    n_files = count_files(dir_reid)
    logger = Logger('./annot_dataset_reid.log')
    data_list = []
    with tqdm(total=n_files, desc=f"Processing {name_processing}") as pbar:
        for root, dirs, files in os.walk(dir_reid):
            for file in files:
                data_list.append((root, file))
                if len(data_list) == batch_size:
                    method_batch(idx_annot, data_list, keys_text, cfg, logger)
                    pbar.update(batch_size)
                    data_list = []
        if len(data_list) != 0:
            method_batch(idx_annot, data_list, keys_text, cfg, logger)
            pbar.update(len(data_list))
        



# def process_smplx_batch(path_cfg):
#     from ..smplx.smplx import render_skeleton
#     cfg = load_cfg(path_cfg)
#     dir_reid = cfg["dir"]["reid"]
#     n_files = count_files(dir_reid)
#     data_list = []
#     with tqdm(total=n_files, desc="Processing get skeleton") as pbar:
#         for root, dirs, files in os.walk(dir_reid):
#             for dir in dirs:
#                 pbar.update(1)
#             for file in files:
#                 if not file.endswith(('.jpg', '.png')):
#                     pbar.update(1)
#                     continue
#                 dir_reid = cfg["dir"]["reid"]
#                 dir_smplx = cfg["dir"]["smplx"]
#                 dir_sub = root[len(dir_reid) + 1:]
                
#                 data_list.append((path_reid, path_pred, path_skeleton))
#                 if len(data_list) == 64:
#                     with ProcessPoolExecutor(max_workers=8) as executor:
#                         list(executor.map(render_skeleton, data_list))
#                     data_list = []
#                     pbar.update(64)
#         with ProcessPoolExecutor(max_workers=8) as executor:
#             list(executor.map(render_skeleton, data_list))
#             pbar.update(len(data_list))    

# def rename_guidance(cfg, root, dir):
#     dir_smplx = cfg["dir"]["smplx"]
#     dir_guidance = os.path.join(dir_smplx, dir, 'smplx')
#     dir_manikin = os.path.join(dir_smplx, dir, 'pred')
#     if os.path.exists(dir_guidance):
#         os.rename(dir_guidance, dir_manikin)
#     else:
#         print("kkkkkk")
#     return
        
 