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
    def __init__(self, dir_sub, name, suff, is_smplx, id_video, 
                    idx_frame, dataset, person, logger, is_check_annot) -> None:
        self._dir = dir_sub
        self._name = name
        self._suff = suff
        self._id_video = id_video
        self._idx_frame = idx_frame
        self._is_smplx = is_smplx
        self._dataset = dataset
        self._person = person
        self._logger = logger
        self._annot = Annotation(
            dir_annot=self.get_dir('annot'),
            path_annot=self.get_path('annot'), 
            img=self,
            logger=self._logger, 
            is_check=is_check_annot,
        )

    def get_annot_dict(self):
        return self._annot

    def get_annot(self, idx):
        return self._annot.get_annot(idx)

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
        return os.path.join(dir_base, self._dir, dir_insert)

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
        path = self.get_path(type)
        if path is None:
            return None
        if not os.path.exists(path):
            return None
        return Image.open(path)

    def get_id_video(self):
        return self._id_video
    
    def get_idx_frame(self):
        return self._idx_frame

    def get_score(self):
        return self._score 

    '''
    is smplx img exists
    '''
    def is_smplx(self):
        return self._is_smplx
    
    '''
    is img belong to a video
    '''
    def is_video(self):
        return self._idx_frame is not None


    '''
    riding has to be the same
    '''
    def is_match_tgt(self, img_tgt):
        if self['riding'] != img_tgt['riding']:
            return False
        if self['hand-carried'] != img_tgt['hand-carried']:
            return False
        self._refresh_score(img_tgt)
        return True

    
    def calib_score(self, img_tgt):
        self._score = float(self['mark_drn'] or 0.0)
        if self['is_riding'] == img_tgt['is_riding']:
            self._score = self._score + 1
        if self['is_hand_carried'] == img_tgt['is_hand_carried']:
            self._score = self._score + 1
        if self['is_backpack'] == img_tgt['is_backpack']:
            self._score = self._score + 1
        if self['color_upper'] == img_tgt['color_upper']:
            self._score = self._score + 1
        if self['color_bottoms'] == img_tgt['color_bottoms']:
            self._score = self._score + 1
        return
    
    def keys(self):
        return self._annot.keys()
    
    def get_key_bool_list(self):
        return self._annot.get_key_bool_list()
    
    def get_key_str_list(self):
        return self._annot.get_key_str_list()

    def rename_key(self, **kwargs):
        key = kwargs['key']
        key_new = kwargs['key_new']
        self._annot.rename_key(key, key_new)

    def remove_key(self, key):
        self._annot.remove_key(key)

    def overwrite_key(self, **kwargs):
        key = kwargs['key']
        data_check = kwargs['data_check']
        data_new = kwargs['data_new']
        self._annot.overwrite_key(key, data_check, data_new)

    def write_key(self, **kwargs):
        key = kwargs['key']
        data = kwargs['data']
        self._annot.write_annot(key, data)

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