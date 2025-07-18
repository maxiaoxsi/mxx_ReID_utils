import random
# import torch
from .set_base import SetBase
from ..object.img import Img
from ..utils.annot.score import add_img_by_score
from tqdm import tqdm

def get_img_standby(img_ref_list, img_sorted_list):
    for img in img_sorted_list:
        if img not in img_ref_list:
            return img
    return None

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
            'infrared': [],
            'visible': [],
        }
        for item in self._list:
            if not item['is_smplx']:
                continue
            self._list_cond['tgt'].append(item)
            if not item['is_visible']:
                self._list_cond['infrared'].append(item)
                continue
            if item['drn'] == 'left':
                self._list_cond['left'].append(item)
            elif item['drn'] == 'right':
                self._list_cond['right'].append(item)
            elif item['drn'] == 'front':
                self._list_cond['front'].append(item)
            elif item['drn'] == 'back':
                self._list_cond['back'].append(item)
        return self._list_cond[cond]

    def get_img_tgt(self, idx_img, stage):
        if isinstance(idx_img, str):
            return self[idx_img]
        if stage in [1, 2, 4]:
            img_list = self.get_list_cond('tgt')
            img_infrared_list = self.get_list_cond('infrared')
            if len(img_infrared_list) > 0:
                img_list = img_infrared_list
        if len(img_list) == 0:
            raise Exception("img_set: img_list empty!")
        if idx_img < 0:
            idx_img = random.randint(0, len(img_list) - 1)
        idx_img = idx_img % len(img_list)
        return img_list[idx_img]

    def get_img_ref_list_from_sorted_list(self, img_tgt, img_sorted_list_drns, is_select_bernl, is_select_repeat):
        img_ref_list = []
        for drn in ["front", "back", "left", "right"]:
            img_sorted_list = img_sorted_list_drns[drn]
            from ..utils.sample.sample import select_img_bernl
            img_ref = select_img_bernl(img_sorted_list, is_select_bernl)
            if not is_select_repeat:
                if img_ref is img_tgt:
                    img_ref = None
                    for img in img_sorted_list:
                        if img is not img_tgt:
                            img_ref = img
                            break
            img_ref_list.append(img_ref)
        return img_ref_list
            


    def get_img_ref_list(self, img_tgt, stage, is_select_bernl, is_select_repeat):
        annot_tgt = img_tgt.annot
        img_sorted_list_drns = {}
        for drn in ["front", "back", "left", "right"]:
            img_sorted_list = self.get_img_sorted_list(annot_tgt, drn)
            img_sorted_list_drns[drn] = img_sorted_list
        img_ref_list = self.get_img_ref_list_from_sorted_list(
            img_tgt=img_tgt, 
            img_sorted_list_drns=img_sorted_list_drns, 
            is_select_bernl=is_select_bernl, 
            is_select_repeat=is_select_repeat,
        )

        if all(x is None for x in img_ref_list):
            img_ref_list = self.get_img_ref_list_from_sorted_list(
                img_tgt=img_tgt, 
                img_sorted_list_drns=img_sorted_list_drns, 
                is_select_bernl=False, 
                is_select_repeat=True,
            )
        
        return img_ref_list, img_sorted_list_drns

    def get_img_sorted_list(self, annot_tgt, drn):
        img_cond_list = self.get_list_cond(drn)
        random.shuffle(img_cond_list)
        img_sorted_list = []
        for img in img_cond_list:
            img.calib_score(annot_tgt)
            add_img_by_score(
                img_list=img_sorted_list, 
                img=img,
            )
        return img_sorted_list

    @property
    def keys(self):
        return self._keys