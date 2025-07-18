import os
from ..annot_base import AnnotBase
from ...ReID.utils.path import get_basename, get_dir_sub, get_path

def annot_vl(idx_annot, data_list, keys_text, cfg, logger):
    annot_list = []
    path_reid_list = []
    texts_annot_list = []
    for (root, file) in data_list:
        dir_sub = get_dir_sub(root, cfg)
        basename, ext = get_basename(file) 
        try:
            if int(file.split(".")[0].split("_")[0]) <= 0:
                continue
        except:
            continue
        path_reid = get_path(cfg, dir_sub, basename, ext, "reid")
        path_annot = get_path(cfg, dir_sub, basename, ext, "annot")
        annot_temp = AnnotBase(path_annot=path_annot, logger=logger)
        texts_annot = []
        for key_text in  keys_text:
            text_annot = f"{key_text}:{annot_temp.get_annot(key_text)}"
            texts_annot.append(text_annot)
        if idx_annot not in annot_temp:
            annot_list.append(annot_temp)
            path_reid_list.append(path_reid)
            texts_annot_list.append(texts_annot)

    if len(annot_list) == 0:
        return
    from ...qwen_vl.qwen import get_annot_batch
    text_list = get_annot_batch(
        path_img_list=path_reid_list,
        texts_annot_list=texts_annot_list,
        idx_annot=idx_annot,
    )
    for annot, text in zip(annot_list, text_list):
        annot.set_annot(idx_annot, text)


# def annot_is_backpack(cfg, root, file, model_qwen, processor_qwen, logger):
#     if not file.endswith(('.jpg', '.png')):
#         return
#     dir_reid = cfg["dir"]["reid"]
#     dir_annot = cfg["dir"]["annot"]
#     dir_sub = root[len(dir_reid) + 1:]
#     name_file = file.split('/')[-1]
#     name_file = file.split('/')[-1]
#     suff_file = name_file.split('.')[-1]
#     name_file = name_file.split('.')[0]
#     path_annot = os.path.join(dir_annot, dir_sub, f'{name_file}.yaml')
#     annot_temp = AnnotBase(path_annot=path_annot, logger=logger)
#     if 'is_backpack_vl' in annot_temp:
#         text = annot_temp.get_annot('is_backpack_vl')
#     else:
#         path_reid = os.path.join(root, file)
#         text = get_is_backpack(
#             model_qwen=model_qwen,
#             processor_qwen=processor_qwen,
#             path_img=path_reid
#         )
#         annot_temp.set_annot('is_backpack_vl', text)
#     if isinstance(text, list):
#         text = text[0]
#     if 'no' in text or text in 'no':
#         annot_temp.set_annot('is_backpack', 'False')
#     elif 'yes' in text or text in 'yes':
#         annot_temp.set_annot('is_backpack', 'True')
#     else:
#         print(text)
#         annot_temp.set_annot('is_backpack', 'True')



# def annot_upper_vl(args):
#     annot_list = []
#     path_reid_list = []
#     # path_annot_list = []
#     for (cfg, root, file, logger) in args:
#         dir_sub = get_dir_sub(root, cfg)
#         basename, ext = get_basename(file) 
#         path_reid = get_path_reid(cfg, dir_sub, basename, ext)
#         path_annot = get_path_annot(cfg, dir_sub, basename)
#         annot_temp = AnnotBase(path_annot=path_annot, logger=logger)
#         if 'upper_vl' not in annot_temp:
#             # path_annot_list.append(path_annot)
#             annot_list.append(annot_temp)
#             path_reid_list.append(path_reid)
#     if len(annot_list) == 0:
#         return
#     from ...qwen_vl.qwen import get_annot_batch
#     text_list = get_annot_batch(
#         path_img_list=path_reid_list,
#         type_prompts="upper_vl",
#     )
#     for annot, text in zip(annot_list, text_list):
#         annot.set_annot('upper_vl', text)