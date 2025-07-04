import torch
import os
import random
from PIL import Image
import yaml
import os
import warnings
import numpy as np
from .annotation import Annotation


class Img:
    def __init__(
        self,
        cache,
        dataset, 
        person, 
        logger, 
    ) -> None:
        
        self._dir_sub = cache["dir_sub"]
        self._name = cache["name"]
        self._suff = cache["suff"]
        # self._is_smplx = cache["is_smplx"]
        self._dataset = dataset
        self._person = person
        self._logger = logger
        self._annot = Annotation(
            path_annot=self.get_path('annot'), 
            img=self,
            logger=self._logger, 
        )

    @property 
    def annot(self):
        return self._annot

    def __getitem__(self, idx):
        return self._annot[idx]

    def __contains__(self, idx):
        return idx in self._annot

    def get_dir(self, tgt):
        dir_base = self._dataset.get_dir(tgt)
        if '_' in tgt:
            dir_insert = tgt.split('_')[-1]
        else:
            dir_insert = ''
        return os.path.join(dir_base, self._dir_sub, dir_insert)

    def get_path(self, tgt):
        dir_tgt = self.get_dir(tgt)
        if tgt == 'annot':
            suff = 'yaml'
        elif tgt == 'smplx_pred':
            suff = 'npz'
        else:
            suff = self._suff
        name = f"{self._name}.{suff}"
        return os.path.join(dir_tgt, name)

    def get_name(self):
        return self._name

    def get_name_img(self):
        return f"{self._name}.{self._suff}"

    def get_img_pil(self, type):
        """Return the image as a PIL Image object."""
        if type in ['background', 'foreground']:
            path_reid = self.get_path("reid")
            path_mask = self.get_path("mask")
            from ...utils.mask import make_back_and_fore_img
            img_fore, img_back = make_back_and_fore_img(
                path_reid=path_reid, 
                path_mask=path_mask
            )
            if type == "background":
                return img_back
            elif type == "foreground":
                return img_fore
        path = self.get_path(type)
        if path is None:
            return None
        if not os.path.exists(path):
            return None
        return Image.open(path)

    @property
    def score(self):
        return self._score 


    def calib_score(self, img_tgt):
        self._score = 0
        for item in [
            'is_riding', 
            'is_hand_carried', 
            'is_backpack', 
            'color_upper', 
            'color_bottoms',
        ]:
            if self[item] == img_tgt[item]:
                self._score += 1
    
    # def keys(self):
    #     return self._annot.keys()
    
    # def get_key_bool_list(self):
    #     return self._annot.get_key_bool_list()
    
    # def get_key_str_list(self):
    #     return self._annot.get_key_str_list()

    # def rename_key(self, **kwargs):
    #     key = kwargs['key']
    #     key_new = kwargs['key_new']
    #     self._annot.rename_key(key, key_new)

    # def remove_key(self, key):
    #     self._annot.remove_key(key)

    # def overwrite_key(self, **kwargs):
    #     key = kwargs['key']
    #     data_check = kwargs['data_check']
    #     data_new = kwargs['data_new']
    #     self._annot.overwrite_key(key, data_check, data_new)

    # def write_key(self, **kwargs):
    #     key = kwargs['key']
    #     data = kwargs['data']
    #     self._annot.write_annot(key, data)

    def get_text_tgt(self):
        text_ref = self.get_text_ref()
        text_ref = text_ref[:len(text_ref) - 1]
        from ..utils.text import get_text_backpack, get_text_hand_carried
        from ..utils.text import get_text_drn
        text_backpack = get_text_backpack(self)
        text_hand_carried = get_text_hand_carried(self)
        text_drn = get_text_drn(self)
        text = f'{text_ref}{text_backpack}{text_hand_carried}{text_drn}.'
        return text

    def get_text_ref(self):
        annot_upper = self["upper"]
        annot_bottoms = self["bottoms"]
        text = f'a photo of a people wearing {annot_upper} and {annot_bottoms}.'
        # a photo of a people wearing red t-shirt and dark shorts, with a backpack,
        return text