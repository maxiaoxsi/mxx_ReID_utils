import os
import numpy as np

from ...ReID.utils.path import get_basename, get_dir_sub, get_path
from ...annot.annot_base import AnnotBase
from ...utils.check import check_is_file_img

def annot_drn(args):
    (cfg, root, file, logger) = args
    if not check_is_file_img(file):
        return
    if int(file.split("_")[0]) <= 0:
        return
    dir_sub = get_dir_sub(root, cfg)
    basename, ext = get_basename(file)
    path_pred = get_path(cfg, dir_sub, basename, ext, "pred")
    if not os.path.exists(path_pred):
        return
    path_annot = get_path(cfg, dir_sub, basename, ext, "annot")
    annot_temp = AnnotBase(path_annot=path_annot, logger=logger)
    if 'drn_smplx' in annot_temp and 'vector_drn_smplx' in annot_temp:
        return
    paras_pred = np.load(path_pred)
    from ...smplx.utils.drn import init_direction
    direction, vector_direction, mark_direction = init_direction(paras_pred)
    annot_temp.set_annot('drn_smplx', direction)
    annot_temp.set_annot('vector_drn_smplx', vector_direction)