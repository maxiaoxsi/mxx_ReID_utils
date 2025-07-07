import os
from ...ReID.utils.path import (get_dir_sub, 
    get_path_reid, get_path_pred, get_path_skeleton
)
from ...utils.path import get_ext

def render_skeleton(args):
    (cfg, root, file, logger) = args
    if not file.endswith(('.jpg', '.png')):
        return
    dir_reid = cfg["dir"]["reid"]
    dir_smplx = cfg["dir"]["smplx"]
    dir_sub = get_dir_sub(root, cfg)
    basename, ext = get_ext(file=file)
    path_reid = get_path_reid(cfg, dir_sub, basename, ext)
    path_pred = get_path_pred(cfg, dir_sub, basename)
    path_skeleton = get_path_skeleton(cfg, dir_sub, basename, ext)
    if not os.path.exists(path_pred):
        return
    if os.path.exists(path_skeleton):
        return
    from ..smplx import render_skeleton
    render_skeleton((path_reid, path_pred, path_skeleton)) 