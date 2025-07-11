import torch
import os
import random
from PIL import Image
import yaml
import os
import warnings
import numpy as np
from .annotation import Annotation
from ..utils.path import get_path



class Img:
    def __init__(
        self,
        basename,
        cache,
        dataset, 
        person, 
        logger, 
    ) -> None:
        self._basename = basename
        self._dir_sub = cache["dir_sub"]
        self._is_smplx = cache["is_smplx"]
        self._dataset = dataset
        self._person = person
        self._logger = logger
        path_annot = get_path(self.dir, self.dir_sub, self.basename, self.ext, "annot")
        self._annot = Annotation(
            path_annot=path_annot, 
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

    def get_img_pil(self, key, rate_mask_aug):
        """Return the image as a PIL Image object."""
        if key in ['mask', 'background', 'foreground']:
            path_reid = get_path(self.dir, self.dir_sub, self.basename, self.ext, "reid")
            path_manikin = get_path(self.dir, self.dir_sub, self.basename, self.ext, "manikin")
            if not os.path.exists(path_manikin):
                return None
            from ...utils.mask import make_mask
            img_mask, img_fore, img_back = make_mask(path_manikin=path_manikin, path_reid=path_reid, rate_mask_aug=rate_mask_aug)
            if key == "mask":
                return img_mask
            if key == "background":
                return img_back
            elif key == "foreground":
                return img_fore
        path = get_path(self.dir, self.dir_sub, self.basename, self.ext, key)
        if not os.path.exists(path):
            if key in ["manikin", "skeleton", "rgbguid"]:
                return None
            print(path)
            raise Exception("path not exists")
        
        return Image.open(path)

    @property
    def score(self):
        return self._score 

    @property
    def dir(self):
        return self._dataset.dir

    @property
    def dir_sub(self):
        return self._dir_sub

    @property
    def basename(self):
        return self._basename

    @property
    def ext(self):
        return self._dataset.ext

    def calib_score(self, annot_tgt):
        self._score = 0
        for item in [
            'is_riding', 
            'is_hand_carried', 
            'is_backpack', 
            'color_upper', 
            'color_bottoms',
        ]:
            if self[item] == annot_tgt[item]:
                self._score += 1
    
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
        is_visible = self["is_visible"]
        annot_upper = self["upper"]
        annot_bottoms = self["bottoms"]
        if is_visible in [True, 'True']:
            str_visible = 'visible'
        elif is_visible in [False, 'False']:
            str_visible = 'infrared'
        else:
            str_visible = 'visible'
            self._logger(f'[img] annot visible wrong: {is_visible}')
        text = f'a {str_visible} photo of a people wearing {annot_upper} and {annot_bottoms}.'
        # a visible photo of a people wearing red t-shirt and dark shorts, with a backpack,
        return text
    




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