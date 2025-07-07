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

def get_dir_base(dir_base, type_dir):
    if isinstance(dir_base, dict):
        if "dir" in dir_base:
            dir_base = dir_base["dir"]
        if type_dir in dir_base:
            dir_base = dir_base[type_dir]
            return dir_base
        else:
            raise Exception("can't load cfg file")
    elif isinstance(dir_base, str):
        return dir_base
    else:
        raise Exception("can't get basename")

def get_path_reid(dir_base, dir_sub, basename, ext):
    dir_base = get_dir_base(dir_base, "reid")
    return os.path.join(dir_base, dir_sub, f"{basename}.{ext}")

def get_path_annot(dir_base, dir_sub, basename):
    dir_base = get_dir_base(dir_base, "annot")
    return os.path.join(dir_base, dir_sub, f"{basename}.yaml")

def get_path_manikin(dir_base, dir_sub, basename, ext):
    dir_base = get_dir_base(dir_base, "smplx")
    return os.path.join(dir_base, dir_sub, 'manikin', f"{basename}.{ext}") 

def get_path_pred(dir_base, dir_sub, basename):
    dir_base = get_dir_base(dir_base, "smplx")
    return os.path.join(dir_base, dir_sub, "pred", f"{basename}.npz")

def get_path_skeleton(dir_base, dir_sub, basename, ext):
    dir_base = get_dir_base(dir_base, "smplx")
    return os.path.join(dir_base, dir_sub, "skeleton", f"{basename}.{ext}")

def load_name(file):
    name_file = file.split('/')[-1]
    suff_file = name_file.split('.')[-1]
    name_file = name_file.split('.')[0]
    return name_file, suff_file

def get_dir_sub(dir, dir_base):
    dir_base = get_dir_base(dir_base, "reid")
    from ...utils.path import get_dir_sub
    return get_dir_sub(dir, dir_base)