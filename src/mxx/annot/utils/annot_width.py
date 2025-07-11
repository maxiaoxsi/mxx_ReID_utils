from PIL import Image

from ...ReID.utils.path import get_basename, get_dir_sub, get_path
from ...annot.annot_base import AnnotBase
from ...utils.check import check_is_file_img

def annot_width(args):
    (cfg, root, file, logger) = args
    if not check_is_file_img(file):
        return
    dir_sub = get_dir_sub(root, cfg)
    basename, ext = get_basename(file)
    path_reid = get_path(cfg, dir_sub, basename, ext, "reid")
    path_annot = get_path(cfg, dir_sub, basename, ext, "annot")
    annot_temp = AnnotBase(path_annot=path_annot, logger=logger)
    width, height = Image.open(path_reid).size
    annot_temp.set_annot("width", width)
    annot_temp.set_annot("height", height)
    return