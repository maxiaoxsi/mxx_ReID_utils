import os
import numpy as np
from tqdm import tqdm
from .dataset import ReIDDataset
from .utils.save_data import save_sample


class ReIDProcessor:
    def __init__(
        self, 
        dataset:ReIDDataset,
        is_vl = False
    ):
        self._dataset = dataset
        if is_vl:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            self._model_qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "/machangxiao/hub/Qwen2.5-VL-7B-Instruct", 
                torch_dtype="auto", 
                # attn_implementation="flash_attention_2",
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

    def test_sample(self, dir, is_norm):
        for i in range(10):
            sample = self._dataset['0002']
            save_sample(
                sample=sample, 
                dir_base=os.path.join(dir, f"img_0_{i}"),
                is_norm=is_norm
            )
            import time
            time.sleep(5)
        


    











    def cluster_image(self, img_1, img_2, key_tgt,
        prompt = "Are the person in the two pictures wearing \
            clothes of the same color? \
            Only answer 'yes' or 'no'", 
    ):
        path_1 = img_1.get_path("reid")
        path_2 = img_2.get_path("reid")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": path_1,
                    },
                    {
                        "type": "image",
                        "image": path_2
                    }, 
                    {
                        "type": "text", 
                        "text": prompt
                    },
                ],
            }
        ]
        output_text = self.get_qwen_annot(messages)
        print(output_text)


                

