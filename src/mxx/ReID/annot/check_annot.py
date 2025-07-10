import os

from ...ReID.utils.path import get_basename, get_dir_sub, get_path
from ...annot.annot_base import AnnotBase
from ...utils.check import check_is_file_img


def check_bool_key(args, key, value_true_list = ['yes', 'true'], 
                   value_false_list = ['no', 'false']):
    key_vl = key + '_vl'
    (cfg, root, file, logger) = args
    if not check_is_file_img(file):
        return
    dir_sub = get_dir_sub(root, cfg)
    basename, ext = get_basename(file)
    path_annot = get_path(cfg, dir_sub, basename, ext, "annot")
    annot_temp = AnnotBase(path_annot=path_annot, logger=logger)
    if key_vl not in annot_temp:
        logger(f"[check_annot] annot miss key {key_vl} in {path_annot}")
        return
    value_vl = annot_temp.get_annot(key_vl).lower()
    if value_vl in value_true_list:
        annot_temp.set_annot(key, "True")
    elif value_vl in value_false_list:
        annot_temp.set_annot(key, "False")
    else:
        logger(f"[check_annot] {path_annot}: key {key_vl} get illegal value {value_vl}.")
    return annot_temp

def check_is_backpack(args):
    check_bool_key(
        args=args,
        key='is_backpack',
    )

def check_is_shoulder_bag(args):
    check_bool_key(
        args=args,
        key='is_shoulder_bag',
    )

def check_is_hand_carried(args):
    check_bool_key(
        args=args,
        key='is_hand_carried',
    )

def check_is_riding(args):
    check_bool_key(
        args=args,
        key='is_riding',
    )

# def check_color(args, key_vl, key_color_vl, key_color):
#     (cfg, root, file, logger) = args
#     dir_sub = get_dir_sub(root, cfg)
#     basename, ext = get_basename(file)
#     path_annot = get_path(cfg, dir_sub, basename, ext, "annot")
#     annot_temp = AnnotBase(path_annot=path_annot, logger=logger)
#     if key_vl not in annot_temp:
#         logger(f"[check_annot] annot miss key {key_vl} in {path_annot}")
#         return
#     value_vl = annot_temp.get_annot(key_vl).lower()
#     words_value_vl = [word for word in value_vl.split() if word]
#     if key_color_vl in annot_temp:
#         value_color_vl = annot_temp.get_annot(key_color_vl).lower()
#         if len(words_value_vl) == 2:
#             annot_temp.set_annot(key_color, words_value_vl[0])
#             return
#         words_value_color_vl = [word for word in value_vl.split() if value_color_vl]
#         for word in words_value_color_vl:
#             if word not in words_value_vl:
#                 logger(f"[check_annot] {path_annot} wrong color_vl")
#         if len(words_value_color_vl) < len(words_value_vl) and words_value_vl[-1] not in words_value_color_vl:
#             annot_temp.set_annot(key_color, value_color_vl)
#             return
#         logger(f"[check_annot] {path_annot} key {key_color_vl} wrong!")
#     else:
#         logger(f"[check_annot] {path_annot} annot miss key {key_color_vl}.")
#         return

# def check_color_upper(args):
#     check_color(
#         args=args,
#         key_vl="upper_vl",
#         key_color_vl="color_upper_vl",
#         key_color="color_upper",
#     )

# def check_color_bottoms(args):
#     check_color(
#         args=args,
#         key_vl="bottoms_vl",
#         key_color_vl="color_bottoms_vl",
#         key_color="color_bottoms",
#     )


