from ast import arg
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

from .set.person_set import PersonSet
from .object.person import Person
from .object.img import Img
from .object.cache import Cache
from ..log.logger import Logger
from .utils.path import load_cfg

class ReIDDataset(Dataset):
    def __init__(
        self,
        path_cfg, # yaml
        path_log="./log.txt",
        is_save=True,
        is_select_bernl=True,
        rate_dropout_ref=0.2,
        rate_dropout_back=0.2,
        rate_dropout_smplx=0.2,
        width_scale=(1, 1),
        height_scale=(1, 1),
        img_size=(512, 512),
        stage = 1,
        n_frame = 10,
    ) -> None:
        self._img_size=img_size
        self._stage=stage
        self._n_frame = n_frame
        self._is_select_bernl = is_select_bernl
        self._rate_dropout_ref = rate_dropout_ref
        self._rate_dropout_back = rate_dropout_back
        self._rate_dropout_smplx = rate_dropout_smplx
        self._logger = Logger(path_log=path_log)
        
        cfg = load_cfg(path_cfg)

        self._dir = cfg['dir']
        self._id = cfg["id_dataset"]

        cache = Cache(
            cfg=cfg,
            logger=self._logger
        )
        self._ext = cache.ext
        self._type = cache.type

        self._person_set = PersonSet(
            dataset=self, 
            logger=self._logger,
        )

        self._person_set.load_cache(
            cache=cache,
        )

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

    def load_img_from_dir(self, dir_person, type_transforms, img_size, type_img = None):
        img_pil_list = []
        if type_img is None:
            dir_img = os.path.join(dir_person, type_transforms)
        else:
            dir_img = os.path.join(dir_person, type_img)
        
        for path in os.listdir(dir_img):
            if not path.endswith((".jpg", ".png", ".jpeg")):
                continue
            path_img = os.path.join(dir_img, path)
            img_pil = Image.open(path_img)
            if len(set(img_pil.getdata())) == 1:
                img_pil = None
            img_pil_list.append(img_pil)
        img_tensor_list = self.get_img_tensor_list(
            img_pil_list=img_pil_list, 
            type_transforms=type_transforms, 
            img_size=img_size,
            seed=None,
        )
        img_tensor = torch.stack(img_tensor_list, dim=0)
        return img_tensor

    def load_text_from_dir(self, dir_person, type_list):
        text_list = []
        dir_text = os.path.join(dir_person, type_list)
        for path in os.listdir(dir_text):
            if not path.endswith(".txt"):
                continue
            path_text = os.path.join(dir_text, path)
            with open(path_text, 'r') as f:
                text = f.read().strip()
            text_list.append(text)
        return text_list

    def load_sample_from_dir(self, dir_sample):
        samples = {}
        for dir in os.listdir(dir_sample):
            dir_person = os.path.join(dir_sample, dir)
            if not os.path.isdir(dir_person):
                continue
            img_reid_tensor = self.load_img_from_dir(
                dir_person=dir_person,
                type_transforms="reid",
                img_size=(128, 256),
            )
            img_ref_tensor = self.load_img_from_dir(
                dir_person=dir_person,
                type_img="reid",
                type_transforms="ref",
                img_size=self._img_size,
            )
            img_background_tensor = self.load_img_from_dir(
                dir_person=dir_person,
                type_transforms="background",
                img_size=self._img_size,
            )
            img_manikin_tensor = self.load_img_from_dir(
                dir_person=dir_person,
                type_transforms="manikin",
                img_size=self._img_size,
            )
            img_skeleton_tensor = self.load_img_from_dir(
                dir_person=dir_person,
                type_transforms="skeleton",
                img_size=self._img_size,
            )
            text_ref_list = self.load_text_from_dir(
                dir_person=dir_person,
                type_list = "reid",
            )
            text_tgt_list = self.load_text_from_dir(
                dir_person=dir_person,
                type_list = "tgt",
            )
            sample = {
                "img_ref_tensor": img_ref_tensor,
                "img_reid_tensor": img_reid_tensor,
                'img_manikin_tensor': img_manikin_tensor,
                'img_skeleton_tensor': img_skeleton_tensor,
                "img_background_tensor": img_background_tensor,
                'text_ref_list': text_ref_list,
                'text_tgt_list': text_tgt_list,
            }
            samples[dir_person] = sample
        return samples   

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
        img_manikin_tensor_list = self.get_img_tensor_list(
            img_pil_list=sample['img_manikin_pil_list'], 
            type_transforms="manikin", 
            seed=seed, 
            img_size=self._img_size
        )
        img_skeleton_tensor_list = self.get_img_tensor_list(
            img_pil_list=sample['img_skeleton_pil_list'], 
            type_transforms="skeleton", 
            seed=seed, 
            img_size=self._img_size
        )
        img_background_tensor_list = self.get_img_tensor_list(
            img_pil_list=sample['img_background_pil_list'], 
            type_transforms="background", 
            seed=seed, 
            img_size=self._img_size
        )

        (
            img_ref_tensor, 
            img_reid_tensor
        ) = self._get_img_tensor(
            rate=self._rate_dropout_ref,
            args_img=(img_ref_tensor_list, img_reid_tensor_list)
        )
        img_tgt_tensor = self._get_img_tensor(None, img_tgt_tensor_list)
        img_manikin_tensor = self._get_img_tensor(None, img_manikin_tensor_list)
        img_skeleton_tensor = self._get_img_tensor(None, img_skeleton_tensor_list)
        img_background_tensor = self._get_img_tensor(
            rate=self._rate_dropout_back,
            args_img=img_background_tensor_list,
        )

        return  {
            "img_ref_tensor": img_ref_tensor,
            "img_reid_tensor": img_reid_tensor,
            "img_tgt_tensor": img_tgt_tensor,
            'img_manikin_tensor': img_manikin_tensor,
            'img_skeleton_tensor': img_skeleton_tensor,
            "img_background_tensor": img_background_tensor,
            'text_ref_list': sample['text_ref_list'],
            'text_tgt_list': sample['text_tgt_list'],
        }
    
    def _get_img_tensor(self, rate, args_img):
        for i in range(len(args_img[0])):
            if rate is not None and random.random() < rate:
                for img_tensor_list in args_img:
                    img_tensor_list[i] = torch.zeros_like(img_tensor_list[i])
        return (torch.stack(item, dim=0) for item in args_img)

    def get_img_tensor_list(self, img_pil_list, type_transforms, img_size, seed = None):
        if type_transforms in ["ref", "tgt", "background", "manikin", "skeleton"]:
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

    def get_dir(self, type_tgt):
        if type_tgt in self._dir:
            return self._dir[type_tgt]
        type_tgt_sub = type_tgt.split('_')[0]
        if type_tgt_sub in self._dir:
            return self._dir[type_tgt_sub]
        print(type_tgt)
        raise Exception(f"dataset:unkown dir type:{type_tgt}")

    def get_person(self, id_person):
        return self._person_set[id_person]

    @property
    def n_frame(self):
        return self._n_frame

    @property 
    def stage(self):
        return self._stage

    @property    
    def keys(self):
        return self._person_set.keys
    
    @property
    def ext(self):
        return self._ext
    
    @property
    def type(self):
        return self._type

    @property
    def dir(self):
        return self._dir