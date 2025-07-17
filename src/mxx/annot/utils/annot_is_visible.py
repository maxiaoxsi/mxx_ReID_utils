import os
import numpy as np

from ...ReID.utils.path import get_basename, get_dir_sub, get_path
from ...annot.annot_base import AnnotBase
from ...utils.check import check_is_file_img

def annot_is_visible(args):
    (cfg, root, file, logger) = args
    if not check_is_file_img(file):
        return
    if int(file.split(".")[0].split("_")[0]) <= 0:
        return
    dir_sub = get_dir_sub(root, cfg)
    basename, ext = get_basename(file)
    path_annot = get_path(cfg, dir_sub, basename, ext, "annot")
    annot_temp = AnnotBase(path_annot=path_annot, logger=logger)
    if 'is_visible' in annot_temp:
        return
    visible_datasets = ['market', 'mars', 'duke', 'msmt17']
    infrared_datasets = ['sysu-mm']
    for name_dataset in visible_datasets:
        if name_dataset in path_annot.lower():
            annot_temp.set_annot('is_visible', True)
            return
    for name_dataset in infrared_datasets:
        if name_dataset in path_annot.lower():
            if 'cam3' in path_annot or 'cam6' in path_annot:
                annot_temp.set_annot('is_visible', False)
            else:
                annot_temp.set_annot('is_visible', True)
            return
    raise Exception("[annot_is_visible] unkown dataset!")