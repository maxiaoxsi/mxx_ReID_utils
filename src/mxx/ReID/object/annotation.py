import os
import yaml
import warnings
import numpy as np
from PIL import Image

'''
annotation for img in ReID datset
_keys: key for img.
    <list: <str>>
    ex: upper_clothing, bottoms, glasses, etc
_path_annot:
_path_log: log file
    <str>
    ex: './log.txt'
_annot: loaded from path_annot
    <dict: {<str>:<str>}>
'''
class Annotation:
    def __init__(self, dir_annot, path_annot, img, logger, is_check) -> None:
        self._keys = []
        self._init_keys()
        self._path_annot = path_annot
        self._img = img
        self._logger = logger
        self._annot = {}
        self._load_annot(dir_annot)

    def get_annot(self, idx):
        if idx in self._annot:
            return self._annot[idx]
        return None

    def __contains__(self, idx):
        return idx in self._annot

    def __getitem__(self, idx):
        idx_smplx = f"{idx}_smplx"
        idx_vl = f"{idx}_vl"
        if idx in self._annot:
            annot = self._annot[idx]
        elif idx_smplx in self._annot:
            annot = self._annot[idx_smplx]
        elif idx_vl in self._annot:
            annot = self._annot[idx_vl]
        else:
            annot = "key error!"
            self._logger.warning(f"annotation: {self._img.get_name()} search key:{idx} not exists in yaml file!")
        if idx in self._key_str_list:
            return annot
        elif idx in self._key_bool_list:
            if annot in ['yes', 'True', 'yes.']:
                return True
            if annot in ['no', 'False', 'no.']:
                return False
            name_reid = self._img.get_name()
            self._logger.warning(f"{name_reid} __getitem__ bool key get other annot:{idx}, {annot}")
            return True

    def get_key_bool_list(self):
        return self._key_bool_list
    
    def get_key_str_list(self):
        return self._key_str_list

    def keys(self):
        return self._keys

    def _init_keys(self):
        self._key_bool_list = [
            "is_backpack", "is_shoulder_bag", "is_hand_carried", 
            "is_visible", "is_riding", 
        ]
        
        self._key_str_list = [
            "upper", "bottoms", 
            "color_upper", "color_bottoms",
            "width", "height", 
            "drn", "vec_drn", "mark_drn",
        ]
        self._keys = self._key_bool_list + self._key_str_list

    def _load_annot(self, dir_annot):
        if os.path.exists(self._path_annot):
            with open(self._path_annot, 'r') as f:
                self._annot = yaml.safe_load(f)
        else:
            if not os.path.exists(dir_annot):
                os.makedirs(dir_annot)
                self._save_annot()
            warnings.warn("mxx object annotation: annotation yaml file not exists, we have ceate empty one")
    
    def _save_annot(self):
        with open(self._path_annot, 'w', encoding='utf-8') as f:
            yaml.safe_dump(
                self._annot, 
                f, 
                allow_unicode=True, 
                default_flow_style=False,
                sort_keys=False 
            )

    def rename_key(self, key, key_new):
        if key not in self._annot:
            name_reid = self._img.get_name
            self._logger.warning(f"{name_reid} rename_key key:{key} miss")
            return
        item = self._annot[key]
        self._annot.pop(key, None)
        self._annot[key_new] = item 
        self._save_annot()

    def remove_key(self, key):
        if key not in self._annot:
            return
        self._annot.pop(key, None)
        self._save_annot()

    def write_annot(self, key, data):
        self._annot[key] = data
        self._save_annot()
        return

    def check_annot(self, key, data):
        if key not in self._annot:
            return False
        return self._annot[key] == data

    def overwrite_key(self, key, data_check, data_new):
        if key not in self._annot:
            return
        if self._annot[key] == data_new:
            return
        if data_check is None:
            self._annot[key] = data_new
            self._save_annot()
            self._logger(f"method: overwrite_key, img: {self._img.get_name()}, key: {key}, data_old: {self._annot[key]}, data_new: {data_new}")
            return
        data_old = self._annot[key]
        if data_check in data_old:
            self._annot[key] = data_new
            self._save_annot()
            self._logger(f"method: overwrite_key, img: {self._img.get_name()}, key: {key}, data_old: {data_old}, data_new: {data_new}")
            return 

    