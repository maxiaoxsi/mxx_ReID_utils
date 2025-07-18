import os
import yaml
import random

def load_cfg(path_cfg, is_check=True):
    from ...utils.path import load_cfg, check_cfg_dir
    cfg = load_cfg(path_cfg)
    if is_check:
        check_cfg_dir(cfg['dir']['reid'])
        check_cfg_dir(cfg['dir']['pred'])

        # check_cfg_dir(cfg['dir']['annot'])
        # check_cfg_dir(cfg['dir']['manikin'])
        # check_cfg_dir(cfg['dir']['annot'])
        # check_cfg_dir(cfg['dir']['mask'])
    return cfg

def get_ext(key, ext):
    if key == 'pred':
        return 'npz'
    elif key == 'annot':
        return 'yaml'
    else:
        return ext

def get_dirname_base(key, dir_base):
    if isinstance(dir_base, dict):
        if "dir" in dir_base:
            dir_base = dir_base["dir"]
        if key in dir_base:
            dir_base = dir_base[key]
            return dir_base
        else:
            raise Exception("can't load cfg file")
    elif isinstance(dir_base, str):
        return dir_base
    else:
        raise Exception("can't get basename")

def get_dirname_rgbguid(dir_base, dir_sub, basename):
    dir_base = get_dirname_base("rgbguid", dir_base)
    return os.path.join(dir_base, dir_sub, basename)

def get_path_rgbguid(dirname_rgbguid):
    from mxx.utils.check import check_is_file_img
    if not os.path.exists(dirname_rgbguid):
        return os.path.join(dirname_rgbguid, "no_result.jpg")
    all_entries = os.listdir(dirname_rgbguid)
    files = [
                f for f in all_entries 
                if os.path.isfile(os.path.join(dirname_rgbguid, f)) 
                and check_is_file_img(os.path.join(dirname_rgbguid, f))
            ]
    if not files:
        return os.path.join(dirname_rgbguid, "no_result.jpg")
    random_file = random.choice(files)
    return os.path.join(dirname_rgbguid, random_file)

def get_path(dir_base, dir_sub, basename, ext, key):
    if key == "rgbguid":
        dirname_rgbguid = get_dirname_rgbguid(dir_base, dir_sub, basename)
        return get_path_rgbguid(dirname_rgbguid)
    dir_base = get_dirname_base(key, dir_base)
    ext = get_ext(key, ext)
    return os.path.join(dir_base, dir_sub, f"{basename}.{ext}")

def get_basename(name_file):
    basename = name_file.split('.')[0]
    ext = name_file.split('.')[-1]
    return basename, ext

def get_dir_sub(dir, dir_base):
    dir_base = get_dirname_base("reid", dir_base)
    from ...utils.path import get_dir_sub
    return get_dir_sub(dir, dir_base)

