import os
import torchvision.transforms as transforms
import shutil
from PIL import Image
import yaml
import random
import torch

def save_item(dataset, id_person, idx_vid, idx_img, dir_base, is_select_bernl):
    person = dataset._person_set[id_person]
    sample = person.get_sample(
        idx_vid=idx_vid,
        idx_img=idx_img,
        n_frame=dataset.n_frame,
        stage=dataset.stage,
        is_select_bernl = is_select_bernl,
    )
    img_ref_pil_list = sample["img_ref_pil_list"]
    img_tgt_pil_list = sample["img_tgt_pil_list"]
    img_manikin_pil_list = sample["img_manikin_pil_list"]
    img_skeleton_pil_list = sample["img_skeleton_pil_list"]
    img_mask_pil_list = sample["img_mask_pil_list"]
    img_foreground_pil_list = sample["img_foreground_pil_list"]
    img_background_pil_list = sample["img_background_pil_list"]
    text_ref_list = sample["text_ref_list"]
    text_tgt_list = sample["text_tgt_list"]
    annot_ref_list = sample["annot_ref_list"]
    annot_tgt_list = sample["annot_tgt_list"]
    save_img_pil(img_ref_pil_list, annot_ref_list, dir_base, "reid")
    save_img_pil(img_tgt_pil_list, annot_tgt_list, dir_base, "tgt")
    save_img_pil(img_manikin_pil_list, annot_tgt_list, dir_base, "manikin")
    save_img_pil(img_skeleton_pil_list, annot_tgt_list, dir_base, "skeleton")
    save_img_pil(img_mask_pil_list, annot_tgt_list, dir_base, "mask")
    save_img_pil(img_background_pil_list, annot_tgt_list, dir_base, "background")
    save_img_pil(img_foreground_pil_list, annot_tgt_list, dir_base, "foreground")
    save_dscrpt_list(annot_ref_list, dir_base, "dscrpt_ref")
    save_dscrpt_list(annot_tgt_list, dir_base, "dscrpt_tgt")
    save_text_list(text_ref_list, dir_base, 'reid', False)
    save_text_list(text_tgt_list, dir_base, 'tgt', False)

def save_sample(sample, dir_base, is_norm):
        img_ref_tensor = sample['img_ref_tensor']
        img_reid_tensor = sample['img_reid_tensor']
        if 'img_tgt_tensor' in sample:
            img_tgt_tensor = sample['img_tgt_tensor']
        img_manikin_tensor = sample['img_manikin_tensor']
        img_skeleton_tensor = sample['img_skeleton_tensor']
        img_background_tensor = sample['img_background_tensor']
        text_ref_list = sample['text_ref_list']
        text_tgt_list = sample['text_tgt_list']
        save_img_tensor(img_ref_tensor, dir_base, "ref", True, is_norm)
        save_img_tensor(img_reid_tensor, dir_base, "reid", True, is_norm)
        if 'img_tgt_tensor' in sample:
            save_img_tensor(img_tgt_tensor, dir_base, "tgt", True, is_norm)
        save_img_tensor(img_ref_tensor, dir_base, "ref", True, is_norm)
        save_img_tensor(img_manikin_tensor, dir_base, "manikin", True, is_norm)
        save_img_tensor(img_skeleton_tensor, dir_base, "skeleton", True, is_norm)
        save_img_tensor(img_background_tensor, dir_base, "background", True, is_norm)
        save_text_list(text_ref_list, dir_base, 'ref', False)
        if 'img_tgt_tensor' not in sample:
            os.makedirs(os.path.join(dir_base, 'tgt'), exist_ok=True)
        save_text_list(text_tgt_list, dir_base, 'tgt', False)

def save_text_list(text_list, dir_base, dir_sub, is_rm):
    dir_save = os.path.join(dir_base, dir_sub)
    if os.path.exists(dir_save) and is_rm:
        shutil.rmtree(dir_save)
    for (i, text) in enumerate(text_list):
        path_save = os.path.join(dir_save, f"text_{i}.txt")
        with open(path_save, 'w') as f:
            f.write(text)

def save_dscrpt_list(dscrpt_list, dir_base, dir_sub, is_clean=True):
    dir_save = os.path.join(dir_base, dir_sub)
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)
    if is_clean:
        import shutil
        shutil.rmtree(dir_save)
        os.makedirs(dir_save)
    for i, dscrpt in enumerate(dscrpt_list):
        path_save = os.path.join(dir_save, f"dscrpt_{i}.yaml")
        with open(path_save, 'w', encoding='utf-8') as f:
            yaml.dump(
                dscrpt, 
                f, 
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False 
            )

def save_img_pil(img_pil_list, dscrpt_list, dir_base, dir_sub, is_clean=True):
    dir_save = os.path.join(dir_base, dir_sub)
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)
    if is_clean:
        import shutil
        shutil.rmtree(dir_save)
        os.makedirs(dir_save)
        
    for (i, (img_pil, dscrpt)) in enumerate(zip(img_pil_list, dscrpt_list)):
        path_img = os.path.join(dir_save, f"img_{i}.jpg")
        if img_pil == None:
            if dscrpt is not None and "width" in dscrpt and "height" in dscrpt:
                width = dscrpt["width"]
                height = dscrpt["height"]
            else:
                width = 64
                height = 128
            img_pil = Image.new('RGB', (width, height), color='black')
        img_pil.save(path_img)

def save_img_tensor(img_tensor, dir_base, dir_sub, is_rm, is_norm):
    dir_save = os.path.join(dir_base, dir_sub)
    if os.path.exists(dir_save) and is_rm:
        shutil.rmtree(dir_save)
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)   
    if (len(img_tensor.shape) == 3):
        img_tensor = img_tensor.squeeze(0)
    if is_norm:
        to_pil = transforms.Compose([
            transforms.Normalize(mean=[-1], std=[2]),
            # transforms.Lambda(lambda x: x[:, :256, :256]),
            transforms.ToPILImage(),
        ])
    else:
        to_pil = transforms.ToPILImage()
    for i in range(img_tensor.shape[0]):
        img = img_tensor[i]
        img_pil = to_pil(img)
        path_img = os.path.join(dir_save, f"img_{i}.jpg")
        img_pil.save(path_img)

def get_annot_list(img_list):
    annot_list = []
    for img in img_list:
        if img is not None:
            annot = img.annot.annot
        else:
            annot = {}
        annot_list.append(annot)
    return annot_list

def get_img_pil_list(img_list, key):
    return [
        img.get_img_pil(key) 
        if img is not None 
        else None
        for img in img_list 
    ]


def load_samples(samples, bs):
    samples_id_person = []
    samples_list = []
    from ...utils import group_by_bs
    for id_person in samples:
        samples_id_person.append(id_person)
        samples_list.append(samples[id_person])
    dir_batch = group_by_bs(samples_id_person, bs)
    sample_batch = group_by_bs(samples_list, bs) 
    samples = []
    for batch in sample_batch:
        img_ref_tensor_list = []
        img_reid_tensor_list = []
        img_manikin_tensor_list = []
        img_skeleton_tensor_list = []
        img_background_tensor_list = []
        text_ref_list = []
        text_tgt_list = []
        for sample in batch:
            img_ref_tensor_list.append(sample['img_ref_tensor'])
            img_reid_tensor_list.append(sample['img_reid_tensor'])
            img_manikin_tensor_list.append(sample['img_manikin_tensor'])
            img_skeleton_tensor_list.append(sample['img_skeleton_tensor'])
            img_background_tensor_list.append(sample['img_background_tensor'])
            for text_ref in sample['text_ref_list']:
                text_ref_list.append(text_ref)
            for text_tgt in sample['text_tgt_list']:
                text_tgt_list.append(text_tgt)
        img_ref_tensor = torch.stack(img_ref_tensor_list, dim=0)
        img_reid_tensor = torch.stack(img_reid_tensor_list, dim=0)
        img_manikin_tensor = torch.stack(img_manikin_tensor_list, dim=0)
        img_skeleton_tensor = torch.stack(img_skeleton_tensor_list, dim=0)
        img_background_tensor = torch.stack(img_background_tensor_list, dim=0)
        sample_batch = {
            'img_ref_tensor': img_ref_tensor,
            'img_reid_tensor': img_reid_tensor,
            'img_manikin_tensor': img_manikin_tensor,
            'img_skeleton_tensor': img_skeleton_tensor,
            'img_background_tensor': img_background_tensor,
            'text_ref_list': text_ref_list,
            'text_tgt_list': text_tgt_list
        }
        samples.append(sample_batch)
    return samples, dir_batch

