import os
import numpy as np

from ...ReID.utils.path import get_basename, get_dir_sub, get_path
from ...annot.annot_base import AnnotBase
from ...utils.check import check_is_file_img

def annot_is_occluded(args):
    (cfg, root, file, logger) = args
    if not check_is_file_img(file):
        return
    
    # if int(file.split(".")[0].split("_")[0]) <= 0:
    #     return
    
    dir_sub = get_dir_sub(root, cfg)
    basename, ext = get_basename(file)
    path_annot = get_path(cfg, dir_sub, basename, ext, "annot")
    annot_temp = AnnotBase(path_annot=path_annot, logger=logger)
    if 'is_occluded' in annot_temp:
        return
    if 'occluded_body_images' in path_annot.lower():
        annot_temp.set_annot('is_occluded', True)
    else:
        annot_temp.set_annot('is_occluded', False)