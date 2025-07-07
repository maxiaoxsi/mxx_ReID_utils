import os
import yaml

def load_cfg(path_cfg, is_check=True):
    if not os.path.exists(path_cfg):
        print(path_cfg)
        raise Exception("dataset cfg file not found!")
    with open(path_cfg, 'r') as f:
        cfg = yaml.safe_load(f)
    if 'dir' not in cfg:
        raise Exception("dir not in dataset cfg file!")
    if is_check:
        check_cfg_dir(cfg['dir']['reid'])
        check_cfg_dir(cfg['dir']['smplx'])
        check_cfg_dir(cfg['dir']['annot'])
        check_cfg_dir(cfg['dir']['mask'])
    return cfg

def check_cfg_dir(dir):
    if not os.path.exists(dir):
        raise Exception(f"dir:{dir} not exists!")

def get_path_annot(dir, sub_dir, name):
    return os.path.join(dir['annot'], sub_dir, f"{name}.yaml")

def get_path_manikin(dir, sub_dir, file):
    return os.path.join(dir["smplx"], sub_dir, 'manikin', file) 