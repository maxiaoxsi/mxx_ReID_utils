from math import log
import os
import time
import random
from torch.utils import data
import yaml
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import pickle

from .set import PersonSet, ImgSet
from .object import Person, Img
from .object.cache import Cache
from ..log.logger import Logger

class ReIDDataset(Dataset):
    def __init__(
        self,
        path_cfg, # yaml
        path_log="./log.txt",
        is_save=True,
        is_check_annot=False,
        is_select_bernl=True,
        rate_dropout_ref=0.2,
        rate_dropout_back=0.2,
        width_scale=(1, 1),
        height_scale=(1, 1),
        img_size_pad=(512, 512),
        stage = 1,
        n_frame = 10,
        is_divide = False,
        st_divide = 0,
        ed_divide = -1,
        n_img = ('', ''),
    ) -> None:
        self._img_size=img_size_pad
        self._stage=stage
        self._n_frame = n_frame
        self._is_select_bernl = is_select_bernl
        self._rate_droupout_ref = rate_dropout_back
        self._rate_droupout_back = rate_dropout_back
        self._logger = Logger(path_log=path_log)

        cfg = self._load_cfg(path_cfg)
        self._dir = cfg['dir']
        self._id = cfg["id_dataset"]
        self._visible = cfg["visible"]

        cache = Cache(
            cfg=cfg,
            logger=self._logger
        )
        self._person_set = PersonSet(dataset=self, logger=self._logger)
        self._person_set.load_cache(cache)

        print(f"load cache from dataset:{self._id}")
        
        self._init_transforms(width_scale, height_scale)


    def __len__(self):
        return len(self._person_set)

    def __getitem__(self, idx):
        return self.get_item(
            id_person=idx,
            idx_img_tgt=-1,
            idx_video_tgt=-1,
        )

    def get_item(self, id_person, idx_img_tgt, idx_video_tgt):
        person = self._person_set[id_person]
        if not isinstance(person, Person):
            return None
        sample = person.get_sample(
            idx_img_tgt=idx_img_tgt,
            idx_video_tgt=idx_video_tgt,
            n_frame=self._n_frame,
            stage=self._stage,
            is_select_bernl=self._is_select_bernl
        )
        seed = int(time.time())
        img_ref_tensor_list = self.get_img_tensor_list(
            img_pil_list=sample['img_ref_pil_list'], 
            type_transforms="ref", 
            img_size=self._img_size,
            seed=seed, 
        )
        img_reid_tensor_list = self.get_img_tensor_list(
            img_pil_list=sample['img_ref_pil_list'], 
            type_transforms="reid", 
            img_size=(128, 256),
            seed=seed
        )
        seed = int(time.time())
        img_tgt_tensor_list = self.get_img_tensor_list(
            img_pil_list=sample['img_tgt_pil_list'], 
            type_transforms="tgt", 
            seed=seed, 
            img_size=self._img_size
        )
        img_smplx_tensor_list = self.get_img_tensor_list(
            img_pil_list=sample['img_smplx_pil_list'], 
            type_transforms="smplx", 
            seed=seed, 
            img_size=self._img_size
        )
        img_background_tensor_list = self.get_img_tensor_list(
            img_pil_list=sample['img_background_pil_list'], 
            type_transforms="background", 
            seed=seed, 
            img_size=self._img_size
        )

        for i in range(len(img_ref_tensor_list)):
            if random.random() < self._rate_droupout_ref:
                img_ref_tensor_list[i] = torch.zeros_like(img_ref_tensor_list[i])
                img_reid_tensor_list[i] = torch.zeros_like(img_reid_tensor_list[i])
        
        for i in range(len(img_background_tensor_list)):
            if random.random() < self._rate_droupout_back:
                img_background_tensor_list[i] = torch.zeros_like(img_background_tensor_list[i])

        img_ref_tensor = torch.stack(img_ref_tensor_list, dim=0)
        img_reid_tensor = torch.stack(img_reid_tensor_list, dim=0)
        img_tgt_tensor = torch.stack(img_tgt_tensor_list, dim=0)
        img_smplx_tensor = torch.stack(img_smplx_tensor_list, dim=0)
        img_background_tensor = torch.stack(img_background_tensor_list, dim=0)

        return  {
            "img_ref_tensor": img_ref_tensor,
            "img_reid_tensor": img_reid_tensor,
            "img_tgt_tensor": img_tgt_tensor,
            'img_smplx_tensor': img_smplx_tensor,
            "img_background_tensor": img_background_tensor,
            'text_ref_list': sample['text_ref_list'],
            'text_tgt_list': sample['text_tgt_list'],
        }
    
    def get_img_tensor_list(self, img_pil_list, type_transforms, img_size, seed = None):
        if type_transforms in ["ref", "tgt", "background", "smplx"]:
            transforms_img=self._transforms_aug_norm_pad
        elif type_transforms == "reid":
            transforms_img=self._transforms_reid
        else:
            raise ValueError(f"Unknown type_transforms: {type_transforms}")
        if seed is not None:
            random.seed(seed)
        w, h = img_size
        img_tensor_list = []
        for img in img_pil_list:
            if img is None:
                img_tensor_list.append(torch.zeros([3, h, w]))
            else:
                img_tensor_list.append(transforms_img(img))
        return img_tensor_list

    def _load_cfg(self, path_cfg):
        if not os.path.exists(path_cfg):
            print(path_cfg)
            raise Exception("dataset cfg file not found!")
        with open(path_cfg, 'r') as f:
            cfg = yaml.safe_load(f)
        if 'dir' not in cfg:
            raise Exception("dir not in dataset cfg file!")
        self._check_cfg_dir(cfg['dir']['reid'])
        self._check_cfg_dir(cfg['dir']['smplx'])
        self._check_cfg_dir(cfg['dir']['annot'])
        self._check_cfg_dir(cfg['dir']['mask'])
        return cfg

    def _check_cfg_dir(self, dir):
        if not os.path.exists(dir):
            raise Exception(f"dir:{dir} not exists!")

    def _analyse_id_dataset(self, dir_reid):
        if "market" in dir_reid.lower():
            return "market"
        if "mars" in dir_reid.lower():
            return "market"
        if "msmt17" in dir_reid.lower():
            return "msmt17"

    def _init_transforms(self, width_scale, height_scale):
        class Scale2D:
            def __init__(self, width, height, interpolation=Image.BILINEAR):
                self.width = width
                self.height = height
                self.interpolation = interpolation
            def __call__(self, img):
                for i in range(4):
                    _ = random.randint(0, 1)
                w, h = img.size
                if h == self.height and w == self.width:
                    return img
                return img.resize((self.width, self.height), self.interpolation)

        class Scale1D:
            def __init__(self, size_tgt, interpolation=Image.BILINEAR):
                self._size_tgt = size_tgt
                self._interpolation = interpolation
            
            def __call__(self, img):
                w, h = img.size
                if w > h:
                    width_tgt = self._size_tgt
                    height_tgt = int(self._size_tgt / w * h)
                else:
                    width_tgt = int(self._size_tgt / h * w)
                    height_tgt = self._size_tgt
                return img.resize((width_tgt, height_tgt), self._interpolation)

        class RandomCrop:
            def __init__(self, width_scale, height_scale):
                self._width_scale = width_scale
                self._height_scale = height_scale

            def _getPoint(self, scale, w):
                w_min, w_max = int(w * scale[0]), int(w * scale[1])
                w_target = random.randint(w_min, w_max)
                w_start = random.randint(0, w - w_target)
                w_end = w_start + w_target
                return w_start, w_end
            
            def __call__(self, img):
                w, h = img.size
                w_start, w_end = self._getPoint(self._width_scale, w)
                h_start, h_end = self._getPoint(self._height_scale, h)
                return img.crop((w_start, h_start, w_end, h_end))

        class PadToBottomRight:
            def __init__(self, target_size, fill=0):
                self.target_size = target_size  # 目标尺寸 (W, H)
                self.fill = fill  # 填充值

            def __call__(self, img):
                """
                img: Tensor of shape [C, H, W]
                Returns: Padded Tensor of shape [C, target_H, target_W]
                """
                _, h, w = img.shape
                pad_w = max(self.target_size[0] - w, 0)  # 右侧需填充的宽度
                pad_h = max(self.target_size[1] - h, 0)  # 底部需填充的高度
                padding = (0, 0, pad_w, pad_h)  
                img_padded = F.pad(img, padding, fill=self.fill)
                return img_padded

        self._transforms_aug_norm_pad = transforms.Compose(
            [
                RandomCrop(width_scale, height_scale),
                Scale1D(self._img_size[0]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
                PadToBottomRight(target_size=self._img_size, fill=0),
            ]
        )

        self._transforms_aug_pad = transforms.Compose(
            [
                RandomCrop(width_scale, height_scale),
                Scale1D(self._img_size[0]),
                transforms.ToTensor(),
                PadToBottomRight(target_size=self._img_size, fill=0),
            ]
        )

        self._transforms_reid=transforms.Compose(
            [
                Scale2D(128, 256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5],std=[0.5]),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            ]
        )

    def get_n_frame(self):
        return self._n_frame
    
    def get_stage(self):
        return self._stage

    def get_dir(self, type_tgt):
        if type_tgt in self._dir:
            return self._dir[type_tgt]
        type_tgt_sub = type_tgt.split('_')[0]
        if type_tgt_sub in self._dir:
            return self._dir[type_tgt_sub]
        raise Exception(f"dataset:unkown dir type:{type_tgt}")

    def get_visible(self):
        return self._visible
    
    def get_person_keys(self):
        return self._person_set.keys
    
    def get_person(self, id_person):
        return self._person_set[id_person]
    
    def get_n_img(self):
        return len(self._img_set)

    def get_img(self, idx):
        return self._img_set[idx]
