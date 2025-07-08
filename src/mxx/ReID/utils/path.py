import os
import yaml

def load_cfg(path_cfg, is_check=True):
    from ...utils.path import load_cfg, check_cfg_dir
    cfg = load_cfg(path_cfg)
    if is_check:
        check_cfg_dir(cfg['dir']['reid'])
        check_cfg_dir(cfg['dir']['smplx'])
        check_cfg_dir(cfg['dir']['annot'])
        check_cfg_dir(cfg['dir']['mask'])
    return cfg

def get_ext(key, ext):
    if key == 'pred':
        return 'npz'
    elif key == 'annot':
        return 'yaml'
    else:
        return ext

def get_dir_base(key, dir_base):
    if isinstance(dir_base, dict):
        if "dir" in dir_base:
            dir_base = dir_base["dir"]
        if key in ["skeleton", "pred", "manikin"]:
            key = "smplx"
        if key in dir_base:
            dir_base = dir_base[key]
            return dir_base
        else:
            raise Exception("can't load cfg file")
    elif isinstance(dir_base, str):
        return dir_base
    else:
        raise Exception("can't get basename")

def get_dir_ext(key):
    if key in ["skeleton", "manikin", "pred"]:
        return key
    else:
        return ""

def get_path(dir_base, dir_sub, basename, ext, key):
    dir_base = get_dir_base(key, dir_base)
    dir_ext = get_dir_ext(key)
    ext = get_ext(key, ext)
    return os.path.join(dir_base, dir_sub, dir_ext, f"{basename}.{ext}")

def get_basename(name_file):
    basename = name_file.split('.')[0]
    ext = name_file.split('.')[-1]
    return basename, ext

def get_dir_sub(dir, dir_base):
    dir_base = get_dir_base("reid", dir_base)
    from ...utils.path import get_dir_sub
    return get_dir_sub(dir, dir_base)

