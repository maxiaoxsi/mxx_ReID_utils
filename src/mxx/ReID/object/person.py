from .img import Img
from ..set.img_set import ImgSet
import random

'''
person in ReID dataset.
person._id: <str> match ReID dataset like 0001, 0002
img_set: <list: <object.img>>, img for person
video_set: <list: <object.video>>, video for person
'''
class Person:
    def __init__(self, id, cache, dataset, logger) -> None:
        self._id = id
        self._cache = cache
        self._img_set = None
        self._dataset = dataset
        self._logger = logger

    def _load_cache(self):
        if self._img_set is not None:
            return
        self._img_set = ImgSet()
        for basename, cache_img in self._cache.items():
            img = Img(
                basename=basename,
                cache=cache_img,
                dataset=self._dataset,
                person=self,
                logger=self._logger
            )
            self._img_set.add_item(basename, img)
        del self._cache

    def __contains__(self, key):
        self._load_cache()
        return key in self._img_set

    def get_sample(self, idx_vid, idx_img, n_frame, stage, is_select_bernl, rate_mask_aug):
        """Get a sample from the person's imgSet or videoSet"""
        """stage1: img, text: visible infrared"""
        """stage2: video"""
        """stage3: infrared only"""
        self._load_cache()
        
        (
            img_tgt_list,
            img_ref_list,
            _,
        )= self.get_img_list(
            idx_vid=idx_vid,
            idx_img=idx_img,
            n_frame=n_frame,
            stage=stage, 
            is_select_bernl=is_select_bernl
        )
        from ..utils.data import get_annot_list, get_img_pil_list
        img_ref_pil_list = get_img_pil_list(img_ref_list, "reid")
        img_tgt_pil_list = get_img_pil_list(img_tgt_list, "reid")
        img_manikin_pil_list = get_img_pil_list(img_tgt_list, "manikin")
        img_skeleton_pil_list = get_img_pil_list(img_tgt_list, "skeleton")
        img_rgbguid_pil_list = get_img_pil_list(img_tgt_list, "rgbguid")
        img_mask_pil_list = get_img_pil_list(img_tgt_list, "mask")
        img_foreground_pil_list = get_img_pil_list(img_tgt_list, "foreground")
        img_background_pil_list = get_img_pil_list(img_tgt_list, "background", rate_mask_aug)
        
        text_ref = img_tgt_list[0].get_text_ref()
        text_tgt = img_tgt_list[0].get_text_tgt()

        annot_ref_list = get_annot_list(img_ref_list)
        annot_tgt_list = get_annot_list(img_tgt_list)

        return {
            'img_ref_pil_list':img_ref_pil_list,
            'img_tgt_pil_list':img_tgt_pil_list,
            'img_manikin_pil_list':img_manikin_pil_list,
            'img_skeleton_pil_list':img_skeleton_pil_list,
            'img_rgbguid_pil_list':img_rgbguid_pil_list,
            'img_mask_pil_list':img_mask_pil_list,
            'img_foreground_pil_list':img_foreground_pil_list,
            'img_background_pil_list':img_background_pil_list,
            'text_ref_list':[text_ref],
            'text_tgt_list':[text_tgt],
            'annot_ref_list':annot_ref_list,
            'annot_tgt_list':annot_tgt_list
        }       

    def get_img_list(self, idx_vid, idx_img, n_frame, stage, is_select_bernl):
        img_tgt_list = self.get_img_tgt_list(
            idx_vid=idx_vid,
            idx_img=idx_img,
            n_frame=n_frame,
            stage=stage,
        )
        annot_tgt = img_tgt_list[0].annot
        
        img_ref_list, img_sorted_list_drns = self.get_img_ref_list(
            annot_tgt=annot_tgt, 
            stage=stage, 
            is_select_bernl=is_select_bernl,
        )
        return img_tgt_list, img_ref_list, img_sorted_list_drns

    def get_img_tgt_list(self, idx_vid, idx_img, n_frame, stage):
        img_tgt = self._img_set.get_img_tgt(
            stage=stage,
            idx_img=idx_img,
        )
        return [img_tgt]
    
    def get_img_ref_list(self, annot_tgt, stage, is_select_bernl):
        return self._img_set.get_img_ref_list(annot_tgt, stage, is_select_bernl)
    
    def __getitem__(self, idx):
        return self._img_set[idx]

    @property
    def id(self):
        return self._id

    @property
    def keys(self):
        self._load_cache()
        return self._img_set.keys
























    

    
    
        
    
    def _get_imgList_stage2(self, num_img_ref = 4, idx_tgt = -1, nframe = -1):
        img_ref_list = self._imgSet.get_img_ref_list(num_img_ref=num_img_ref, stage=1)
        img_tgt_list = self._videoSet.get_img_tgt_list(idx_tgt=idx_tgt, nframe=nframe)
        if random.random() < 0.5:
            img_ref_list.append(img_tgt_list[0])
        else:
            img_ref_list.append(None)
        return img_ref_list, img_tgt_list

    def _get_imgList_stage3(self, num_img_ref = 4, idx_tgt = -1):
        img_ref_list = self._imgSet.get_img_ref_list(
            stage=3,
            num_img_ref=num_img_ref,
        )
        img_tgt_list = self._imgSet.get_img_tgt_list(
            stage = 3,
            idx_tgt = idx_tgt,
        )
        return img_ref_list, img_tgt_list
        