import os
import numpy as np
from tqdm import tqdm
from .dataset import ReIDDataset
from .utils.save_data import save_sample


class ReIDProcessor:
    def __init__(
        self, 
        dataset=None,
        is_vl=False
    ):
        self._dataset = None
        self._dataset_list = []
        if dataset is not None:
            self._dataset = dataset
            self._dataset_list.append(dataset)
        if is_vl:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            self._model_qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "/machangxiao/hub/Qwen2.5-VL-7B-Instruct", 
                torch_dtype="auto", 
                device_map="cuda"
            )

            self._processor_qwen = AutoProcessor.from_pretrained(
                "/machangxiao/hub/Qwen2.5-VL-7B-Instruct", 
                use_fast=False
            )

        
    def __call__(self, method_name, **kwargs):
        for i in tqdm(range(self._dataset.get_n_img())):
            img = self._dataset.get_img(i)
            method = getattr(img, method_name)
            method(**kwargs)
 
    def get_qwen_annot(self, messages):
        text = self._processor_qwen.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor_qwen(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model_qwen.device)
        generated_ids = self._model_qwen.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._processor_qwen.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text

    def get_messages(self, img_list, img_tgt, key_annot_list, prompt):
        content_list = []
        from .object.img import Img
        if img_list is not None:
            for img in img_list:
                if isinstance(img, Img):
                    path_img = img.get_path("reid")
                elif isinstance(img, str):
                    path_img = img
                content_list.append(
                    {
                        "type": "image",
                        "image": path_img,
                    }
                )
        if img_tgt is not None: 
            for key_annot in key_annot_list:
                if key_annot in img_tgt:
                    content_list.append(
                        {
                            "type": "text",
                            "text": f"{key_annot}:{img_tgt.get_annot(key_annot)}",
                        }
                    )
        content_list.append(
            {
                "type": "text",
                "text": prompt
            }
        )
        messages = [
            {
                "role": "user",
                "content": content_list
            }
        ]
        return messages

    def get_annot_img(self, key_tgt, prompt):
        for i in tqdm(range(self._dataset.get_num_img())):
            img = self._dataset.get_img(i)
            messages = self.get_messages(
                img_list=[img],
                img_tgt=None,
                key_annot_list=None,
                prompt=prompt
            )
            output_text = self.get_qwen_annot(messages)
            img.write_annot(key_tgt, output_text[0].lower())

    def get_annot_annot(self, key_annot_list, key_tgt, prompt):
        for i in tqdm(range(self._dataset.get_num_img())):
            img = self._dataset.get_img(i)
            messages = self.get_messages(
                img_list=[],
                img_tgt=img,
                key_annot_list = key_annot_list,
                prompt=prompt
            )
            # print(messages)
            output_text = self.get_qwen_annot(messages)
            # print(output_text)
            # exit()
            img.write_annot(key_tgt, output_text[0].lower())
        
    def get_skeleton(self):
        from ..smplx.painter import Painter
        painter_mxx = Painter(
            path_smplx_model='/machangxiao/code/smplx/models'
        )
        for i in tqdm(range(self._dataset.get_n_img())):
            img = self._dataset.get_img(i)
            painter_mxx(img=img, is_save_skeleton=True, is_save_manikin=True)

    def sort_and_divide_cache(self, dir_cache, name_base, num_divided):
        import pickle
        path_cache = os.path.join(dir_cache, f'{name_base}.pkl')
        with open(path_cache, 'rb') as f:
            data = pickle.load(f)
        data['img'] = sorted(data['img'], key=lambda x: x['id_person'])
        len_slice = len(data['img']) // num_divided
        idx_st = 0
        for i in range(num_divided):
            if i == num_divided - 1:
                idx_end = len(data['img'])
            else:
                idx_end = idx_st + len_slice
                while idx_end + 1 < len(data['img']) and (data['img'][idx_end]['id_person'] == data['img'][idx_end + 1]['id_person']):
                    idx_end += 1
                idx_end += 1
            data_divided = data['img'][idx_st:idx_end]
            data_save = {'img':data_divided}
            with open(os.path.join(dir_cache, f"{name_base}_{i}.pkl") , 'wb') as f:
                pickle.dump(data_save, f)
            if (idx_end == len(data['img'])):
                print(f'{i} is the last one')
                break
            idx_st = idx_end
            









                

