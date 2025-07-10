import os
from ...ReID.utils.path import get_dir_sub, get_basename, get_path
from ...utils.check import check_is_file_img

def render_skeleton(args):
    (cfg, root, file, logger) = args
    if not check_is_file_img(file=file):
        return
    dir_sub = get_dir_sub(root, cfg)
    basename, ext = get_basename(name_file=file)
    path_reid = get_path(cfg, dir_sub, basename, ext, "reid")
    path_pred = get_path(cfg, dir_sub, basename, ext, "pred")
    path_skeleton = get_path(cfg, dir_sub, basename, ext, "skeleton")
    if not os.path.exists(path_pred):
        return
    if os.path.exists(path_skeleton):
        return
    from ..smplx import render_skeleton
    render_skeleton((path_reid, path_pred, path_skeleton)) 