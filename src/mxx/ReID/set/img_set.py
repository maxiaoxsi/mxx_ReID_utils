import random
# import torch
from .set_base import SetBase
from ..object import Img
from ..utils.annot.score import add_img_by_score
from tqdm import tqdm



class ImgSet(SetBase):
    def __init__(self) -> None:
        super().__init__()
        self._list_cond = None
        self._keys = []

    def add_item(self, key, item):
        self._keys.append(key)
        return super().add_item(key, item)
    
    def get_list_cond(self, cond):
        if self._list_cond is not None:
            return self._list_cond[cond]
        self._list_cond = {
            'front': [],
            'back': [],
            'left': [],
            'right': [],
            'tgt': [],
        }
        for item in self._list:
            if not item['is_smplx']:
                continue
            self._list_cond['tgt'].append(item)
            if item['drn'] == 'left':
                self._list_cond['left'].append(item)
            elif item['drn'] == 'right':
                self._list_cond['right'].append(item)
            elif item['drn'] == 'front':
                self._list_cond['front'].append(item)
            elif item['drn'] == 'back':
                self._list_cond['back'].append(item)
        return self._list_cond[cond]

    def get_img_tgt(self, idx_img_tgt, stage):
        if isinstance(idx_img_tgt, str):
            return self[idx_img_tgt]
        if stage in [1, 2, 4]:
            img_list = self.get_list_cond('tgt')
        elif stage in [3]:
            img_list = self.get_list_cond('infrared')
        if len(img_list) == 0:
            raise Exception("img_set: img_list empty!")
        if idx_img_tgt < 0:
            idx_img_tgt = random.randint(0, len(img_list) - 1)
        idx_img_tgt = idx_img_tgt % len(img_list)
        return img_list[idx_img_tgt]

    def get_img_ref(self, img_tgt, stage, is_select_bernl):
        img_ref_list = []
        list_img_sorted_dict = {}
        for drn in ["front", "back", "left", "right"]:
            img_sorted_list = self.get_img_sorted_list(img_tgt, drn)
            list_img_sorted_dict[drn] = img_sorted_list
            from ..utils.sample.sample import select_img_bernl
            img_ref = select_img_bernl(img_sorted_list, is_select_bernl)
            img_ref_list.append(img_ref)
        return img_ref_list, list_img_sorted_dict

    def get_img_sorted_list(self, img_tgt, drn):
        img_cond_list = self.get_list_cond(drn)
        img_sorted_list = []
        for img in img_cond_list:
            img.calib_score(img_tgt)
            add_img_by_score(
                img_list=img_sorted_list, 
                img=img,
            )
            # print(len(img_cond_list))
        return img_sorted_list

    @property
    def keys(self):
        return self._keys