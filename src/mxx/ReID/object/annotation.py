from math import log
import os
import yaml
import warnings
import numpy as np
from PIL import Image
from ...annot.annot_base import AnnotBase
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
class Annotation(AnnotBase):
    def __init__(self, path_annot, img, logger) -> None:
        super().__init__(path_annot=path_annot, logger=logger)
        self._keys = []
        self._init_keys()
        self._img = img

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
            self._logger.warning(f"annotation: {self._img.basename} search key:{idx} not exists in yaml file!")
        if idx in self._key_str_list:
            return annot
        elif idx in self._key_bool_list:
            if annot in ['yes', 'True', 'yes.']:
                return True
            if annot in ['no', 'False', 'no.']:
                return False
            name_reid = self._img.basename
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
            "is_backpack", 
            "is_shoulder_bag", 
            "is_hand_carried", 
            "is_visible", 
            "is_riding", 
            "is_smplx"
        ]
        
        self._key_str_list = [
            "upper", 
            "bottoms", 
            "color_upper", 
            "color_bottoms",
            "width", 
            "height", 
            "drn", 
            "vec_drn", 
            "mark_drn",
        ]
        self._keys = self._key_bool_list + self._key_str_list

    def rename_key(self, key, key_new):
        if key not in self._annot:
            name_reid = self._img.get_name
            self._logger.warning(f"{name_reid} rename_key key:{key} miss")
            return
        item = self._annot[key]
        self._annot.pop(key, None)
        self._annot[key_new] = item 
        self._save_annot()
