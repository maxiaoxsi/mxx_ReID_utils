import os
import yaml

def load_cfg(path_cfg):
    if not os.path.exists(path_cfg):
        print(path_cfg)
        raise Exception("dataset cfg file not found!")
    with open(path_cfg, 'r') as f:
        cfg = yaml.safe_load(f)
    if 'dir' not in cfg:
        raise Exception("dir not in dataset cfg file!")
    return cfg

def check_cfg_dir(dir):
    if not os.path.exists(dir):
        raise Exception(f"dir:{dir} not exists!")
    
def get_dir_sub(dir, dir_base):
    if not os.path.isdir(dir):
        raise Exception("dir not a dir")
    if not os.path.isdir(dir_base):
        raise Exception("dir_base not a dir")
    return dir[len(dir_base) + 1:]

def get_basename(name_file):
    basename = name_file.split('.')[0]
    ext = name_file.split('.')[-1]
    return basename, ext
