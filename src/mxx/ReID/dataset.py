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
        is_select_repeat=True,
        rate_random_erase=0.5,
        rate_dropout_ref=0.2,
        rate_dropout_back=0.2,
        rate_dropout_manikin=0,
        rate_dropout_skeleton=0.2,
        rate_dropout_rgbguid=1,
        rate_mask_aug = 1,
        width_scale=(1, 1),
        height_scale=(1, 1),
        img_size=(512, 512),
        stage = 1,
        n_frame = 10,
    ) -> None:
        self._img_size=img_size
        self._img_size_reid=(128, 256)
        self._stage=stage
        self._n_frame = n_frame
        self._is_select_bernl = is_select_bernl
        self._is_select_repeat = is_select_repeat
        self._rate_random_erase = rate_random_erase
        self._rate_dropout_ref = rate_dropout_ref
        self._rate_dropout_back = rate_dropout_back
        self._rate_dropout_manikin = rate_dropout_manikin
        self._rate_dropout_skeleton = rate_dropout_skeleton
        self._rate_dropout_rgbguid = rate_dropout_rgbguid
        self._rate_mask_aug = rate_mask_aug
        self._logger = Logger(path_log=path_log)
        
        cfg = load_cfg(path_cfg)

        self._dir = cfg['dir']
        self._id = cfg["id_dataset"]

        cache = Cache(
            cfg=cfg,
            logger=self._logger,
            is_save=is_save,
        )
        self._ext = cache.ext
        self._type = cache.type

        self._person_set = PersonSet(
            dataset=self, 
            logger=self._logger,
            cache=cache,
        )

        print(f"load cache from dataset:{self._id}")
        
        self._init_transforms(width_scale, height_scale)


    def __len__(self):
        return len(self._person_set)

    def __getitem__(self, idx):
        return self.get_item(
            id_person=idx,
            idx_vid=-1,
            idx_img=-1,
        )
   
    def __contains__(self, key):
        return key in self.keys

    def get_item(self, id_person, idx_vid, idx_img):
        person = self._person_set[id_person]
        if not isinstance(person, Person):
            return None
        sample = person.get_sample(
            idx_vid=idx_vid,
            idx_img=idx_img,
            n_frame=self.n_frame,
            stage=self.stage,
            is_select_bernl = self.is_select_bernl,
            is_select_repeat = self._is_select_repeat,
            rate_mask_aug = self.rate_mask_aug,
        )
        seed = int(time.time())
        img_ref_tensor_list = self.get_img_tensor_list(
            img_pil_list=sample['img_ref_pil_list'], 
            type_transforms="ref", 
            img_size=self._img_size,
            is_train=True,
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
            img_size=self._img_size,
        )
        img_skeleton_tensor_list = self.get_img_tensor_list(
            img_pil_list=sample['img_skeleton_pil_list'], 
            type_transforms="skeleton", 
            seed=seed, 
            img_size=self._img_size,
        )
        img_rgbguid_tensor_list = self.get_img_tensor_list(
            img_pil_list=sample['img_rgbguid_pil_list'],
            type_transforms="rgbguid",
            seed=seed,
            img_size=self._img_size,
        )
        img_background_tensor_list = self.get_img_tensor_list(
            img_pil_list=sample['img_background_pil_list'], 
            type_transforms="background", 
            seed=seed, 
            img_size=self._img_size,
        )
        for i in range(len(img_background_tensor_list)):
            if i != 0:
                img_background_tensor_list[i] = torch.zeros_like(img_background_tensor_list[i])
                

        (
            img_ref_tensor, 
            img_reid_tensor
        ) = self._get_img_tensor(
            rate=self._rate_dropout_ref,
            args_img=(img_ref_tensor_list, img_reid_tensor_list)
        )
        (img_tgt_tensor, ) = self._get_img_tensor(
            rate=None, 
            args_img=(img_tgt_tensor_list, ))
        (img_manikin_tensor, ) = self._get_img_tensor(
            rate=self._rate_dropout_manikin, 
            args_img=(img_manikin_tensor_list, ),
        )
        (img_skeleton_tensor, ) = self._get_img_tensor(
            rate=self._rate_dropout_skeleton, 
            args_img=(img_skeleton_tensor_list, ),
        )
        (img_rgbguid_tensor, ) = self._get_img_tensor(
            rate=self._rate_dropout_rgbguid,
            args_img=(img_rgbguid_tensor_list, ),
        )
        (img_background_tensor, ) = self._get_img_tensor(
            rate=self._rate_dropout_back,
            args_img=(img_background_tensor_list, ),
        )

        return  {
            "img_ref_tensor": img_ref_tensor,
            "img_reid_tensor": img_reid_tensor,
            "img_tgt_tensor": img_tgt_tensor,
            'img_manikin_tensor': img_manikin_tensor,
            'img_skeleton_tensor': img_skeleton_tensor,
            'img_rgbguid_tensor': img_rgbguid_tensor,
            "img_background_tensor": img_background_tensor,
            'text_ref_list': sample['text_ref_list'],
            'text_tgt_list': sample['text_tgt_list'],
        }
    
    def _get_img_tensor(self, rate, args_img):
        for i in range(len(args_img[0])):
            if rate is not None and random.random() < rate:
                for img_tensor_list in args_img:
                    img_tensor_list[i] = torch.zeros_like(img_tensor_list[i])
        return tuple(torch.stack(item, dim=0) for item in args_img)

    def get_img_tensor_list(self, img_pil_list, type_transforms, 
            img_size, is_train=False, seed = None):
        if type_transforms in ["ref", "tgt", "background", "manikin", "skeleton", "rgbguid"]:
            if is_train and type_transforms == "ref":
                transforms_img=self._transforms_ref
            else:
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

        self._transforms_ref = transforms.Compose(
            [
                RandomCrop(width_scale, height_scale),
                Scale1D(self._img_size[0]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
                transforms.RandomErasing(p=self._rate_random_erase, scale=(0.12, 0.37), ratio=(0.3, 3.3), value=0, inplace=False),
                PadToBottomRight(target_size=self._img_size, fill=0),
            ]
        )

        self._transforms_reid=transforms.Compose(
            [
                Scale2D(128, 256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5],std=[0.5]),
            ]
        )

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
    def img_size(self):
        return self._img_size

    @property
    def dir(self):
        return self._dir
    
    @property
    def is_select_bernl(self):
        return self._is_select_bernl

    @property
    def rate_mask_aug(self):
        return self._rate_mask_aug

    def load_sample_pil_from_dir(self, dir_person):
        img_reid_pil_list = self.load_img_pil_from_dir(
            dir_person=dir_person,
            dir_sub="reid",
        )
        img_ref_pil_list = self.load_img_pil_from_dir(
            dir_person=dir_person,
            dir_sub="reid",
        )
        img_background_pil_list = self.load_img_pil_from_dir(
            dir_person=dir_person,
            dir_sub="background",
        )
        img_manikin_pil_list = self.load_img_pil_from_dir(
            dir_person=dir_person,
            dir_sub="manikin",
        )
        img_skeleton_pil_list = self.load_img_pil_from_dir(
            dir_person=dir_person,
            dir_sub="skeleton",
        )
        img_rgbguid_pil_list = self.load_img_pil_from_dir(
            dir_person=dir_person,
            dir_sub="rgbguid",
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
            "img_ref_pil_list": img_ref_pil_list,
            "img_reid_pil_list": img_reid_pil_list,
            'img_manikin_pil_list': img_manikin_pil_list,
            'img_skeleton_pil_list': img_skeleton_pil_list,
            "img_rgbguid_pil_list": img_rgbguid_pil_list,
            "img_background_pil_list": img_background_pil_list,
            'text_ref_list': text_ref_list,
            'text_tgt_list': text_tgt_list,
        }
        return sample
        
    def load_samples_pil_dict_from_dir(self, dir_sample, n_max=-1):
        samples = {}
        i = 0
        for dir in os.listdir(dir_sample):
            if n_max != -1 and i > n_max:
                return samples
            dir_person = os.path.join(dir_sample, dir)
            if not os.path.isdir(dir_person):
                continue
            sample = self.load_sample_pil_from_dir(dir_person)
            samples[dir_person] = sample
            i = i + 1
        return samples

    def get_sample_tensor_from_sample_pil(self, sample):
        img_ref_tensor = self.get_img_tensor(
            img_pil_list=sample["img_ref_pil_list"],
            type_transforms="ref",
            img_size=self._img_size,
        )
        img_reid_tensor = self.get_img_tensor(
            img_pil_list=sample["img_reid_pil_list"],
            type_transforms="reid",
            img_size=self._img_size_reid,
        )
        img_manikin_tensor = self.get_img_tensor(
            img_pil_list=sample['img_manikin_pil_list'],
            type_transforms="manikin",
            img_size=self._img_size,
        )
        img_skeleton_tensor = self.get_img_tensor(
            img_pil_list=sample['img_skeleton_pil_list'],
            type_transforms="skeleton",
            img_size=self._img_size,
        )
        img_rgbguid_tensor = self.get_img_tensor(
            img_pil_list=sample['img_rgbguid_pil_list'],
            type_transforms="rgbguid",
            img_size=self._img_size,
        )
        img_background_tensor = self.get_img_tensor(
            img_pil_list=sample['img_background_pil_list'],
            type_transforms="background",
            img_size=self._img_size,
        )
        sample_tensor = {
            'img_ref_tensor': img_ref_tensor,
            'img_reid_tensor': img_reid_tensor,
            'img_manikin_tensor': img_manikin_tensor,
            'img_skeleton_tensor': img_skeleton_tensor,
            'img_rgbguid_tensor': img_rgbguid_tensor,
            'img_background_tensor': img_background_tensor,
            'text_ref_list': sample['text_ref_list'],
            'text_tgt_list': sample['text_tgt_list'],
        }
        return sample_tensor

    def get_samples_tensor_dict(self, samples_pil_dict):
        samples_tensor = {}
        for key, sample in samples_pil_dict.items():
            sample_tensor = self.get_sample_tensor_from_sample_pil(sample)
            samples_tensor[key] = sample_tensor
        return samples_tensor
            
    def load_sample_from_dir(self, dir_sample, n_max = -1):
        # samples = {}
        samples_pil = self.load_samples_pil_dict_from_dir(dir_sample, n_max)
        samples_tensor = self.get_samples_tensor_dict(samples_pil)
        return samples_tensor

    def load_img_tensor_from_pil(self, img_pil_list, type_transforms, img_size=(128, 256)):
        img_tensor_list = self.get_img_tensor_list(
            img_pil_list=img_pil_list, 
            type_transforms=type_transforms, 
            img_size=img_size,
            seed=None,
        )
        img_tensor = torch.stack(img_tensor_list, dim=0)
        return img_tensor

    def load_img_tensor_from_path(self, path_list, type_transforms, img_size=(128, 256)):
        img_pil_list = []
        for path_img in path_list:
            if not os.path.exists(path_img):
                img_pil = None
            else:
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

    def load_img_pil_from_dir(self, dir_person, dir_sub):
        img_pil_list = []
        dir_img = os.path.join(dir_person, dir_sub)
        for path in os.listdir(dir_img):
            if not path.endswith((".jpg", ".png", ".jpeg")):
                continue
            path_img = os.path.join(dir_img, path)
            img_pil = self.load_img_pil_from_path(path_img)
            img_pil_list.append(img_pil)
        return img_pil_list

    def load_img_pil_from_path(self, path_img):
        img_pil = Image.open(path_img)
        if len(set(img_pil.getdata())) == 1:
                img_pil = None
        return img_pil

        

    def get_img_tensor(self, img_pil_list, type_transforms, img_size):
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

    

