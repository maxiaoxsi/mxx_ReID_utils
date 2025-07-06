import os
from ...utils.batch import load_cfg
from ...utils.batch import load_name

def render_skeleton(args):
    (cfg, root, file, logger) = args
    if not file.endswith(('.jpg', '.png')):
        return
    dir_reid = cfg["dir"]["reid"]
    dir_smplx = cfg["dir"]["smplx"]
    dir_sub = root[len(dir_reid) + 1:]
    name_file, suff_file = load_name(file=file)
    path_reid = os.path.join(root, file)
    path_pred = os.path.join(dir_smplx, 
        dir_sub, 'pred', f'{name_file}.npz')
    path_skeleton = os.path.join(dir_smplx,
        dir_sub, 'skeleton', f'{name_file}.{suff_file}')
    if not os.path.exists(path_pred):
        return
    if os.path.exists(path_skeleton):
        return
    from ..smplx import render_skeleton
    render_skeleton((path_reid, path_pred, path_skeleton)) 