from math import log
import os
import yaml
from ..annot.annot_base import AnnotBase
from ..log.logger import Logger

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


def set_annot_smplx_batch(path_cfg = None, dir_reid=None, dir_annot=None):
    if path_cfg is not None:
        cfg = load_cfg(path_cfg)
        dir_reid = cfg["dir"]["reid"]
        dir_annot = cfg["dir"]["annot"]
        dir_smplx = cfg["dir"]["smplx"]
    for root, dirs, files in os.walk(dir_reid):
        dir_sub = root[len(dir_reid) + 1:]
        for file in files:
            if not file.endswith(('.jpg', '.png')):
                continue
            name_file = file.split('/')[-1]
            name_file = file.split('/')[-1]
            suff_file = name_file.split('.')[-1]
            name_file = name_file.split('.')[0]
            logger = Logger('./batch.log')
            path_smplx_guid = os.path.join(dir_smplx, 
                dir_sub, 'manikin', f'{name_file}.{suff_file}')
            path_annot = os.path.join(dir_annot, dir_sub, f'{name_file}.yaml')
            annot_temp = AnnotBase(path_annot=path_annot, logger=logger)
            if os.path.exists(path_smplx_guid):
                annot_temp.set_annot('is_smplx', 'True')
            else:
                annot_temp.set_annot('is_smplx', 'False')

