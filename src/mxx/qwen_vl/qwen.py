import glob

model_qwen = None
processor_qwen = None

def load_qwen():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model_qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "/machangxiao/hub/Qwen2.5-VL-7B-Instruct", 
        torch_dtype="auto", 
        device_map="cuda"
    )

    processor_qwen = AutoProcessor.from_pretrained(
        "/machangxiao/hub/Qwen2.5-VL-7B-Instruct", 
        use_fast=False
    )
    return model_qwen, processor_qwen

def init_prompts():
    prompts = {}

    prompts["bottoms_vl"] = "Could you please describe \
the color and style of the bottoms of the people? \
Answer with a brief phrase 'color' + 'style'\
such as  'black jeans'. \
A brief phrase is enough, no full sentences."
    
    prompts["upper_vl"] = "Could you please describe \
the color and style of the upper clothing of the people? \
Answer with a brief phrase 'color' + 'style'\
such as  'white t-shirt'. \
A brief phrase is enough, no full sentences."

    prompts["color_upper_vl"] = "Could you please describe \
the color of the upper clothing from the text phrase upper_vl? \
Answer with a brief phrase 'color' from phrase 'color' + 'style' \
such as answer 'white' when the text is 'upper_vl:white t-shirt'. \
A brief phrase is enough, no full sentences."

    prompts["color_bottoms_vl"] = "Could you please describe \
the color of the bottoms from the text phrase bottoms_vl?\
Answer with a brief phrase 'color' from phrase 'color' + 'style'\
such as answer 'red' when the text is 'bottoms_vl:red shorts'. \
A brief phrase is enough, no full sentences." 

    prompts["is_shoulder_bag_vl"] = "Could you please describe \
is the person in the photo carrying a shoulder bag?\
please anwser with a brief phrase 'yes' or 'no'"

    prompts["is_backpack_vl"] = "Could you please describe \
is the person in the photo wearing a backpack?\
please anwser with a brief phrase 'yes' or 'no'"
    
    prompts["is_hand_carried_vl"] = "Could you please describe \
is the person in the photo carrying some object in his or her hand?\
Attention that only the object carried on his hand,\
the backpack and shoulder bag not count!\
please anwser with a brief phrase 'yes' or 'no'"

    return prompts

def get_qwen_annot(messages_list):
    texts_batch = []
    images_batch = []
    videos_batch = []
    for messages in messages_list:
        text = processor_qwen.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        texts_batch.append(text)
        images_batch.append(image_inputs)
        videos_batch.append(video_inputs)
    inputs = processor_qwen(
        text=texts_batch,
        images=images_batch if any(images_batch) else None,
        videos=videos_batch if any(videos_batch) else None,
        padding=True,
        padding_side='left',
        return_tensors="pt",
    )
    inputs = inputs.to(model_qwen.device)
    generated_ids = model_qwen.generate(
        **inputs, 
        max_new_tokens=128,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor_qwen.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
    return output_text

def get_messages(path_img_list, text_annot_list, prompt_list):
        content_list = []
        if path_img_list is not None:
            for path_img in path_img_list:
                content_list.append(
                    {
                        "type": "image",
                        "image": path_img,
                    }
                )
        if text_annot_list is not None:
            for text_annot in text_annot_list:
                content_list.append(
                    {
                        "type":"text",
                        "text":text_annot,
                    }
                )
        for prompt in prompt_list:
            content_list.append(
                {
                    "type": "text",
                    "text": prompt,
                }
            )

        messages = [
            {
                "role": "user",
                "content": content_list
            }
        ]
        return messages

def get_annot_img(path_img_list, texts_annot_list, prompt):
    messages_list = []
    for path_img, texts_annot in zip(path_img_list, texts_annot_list):
        messages = get_messages(
            path_img_list=[path_img],
            text_annot_list = texts_annot,
            prompt_list=[prompt]
        )
        messages_list.append(messages)
    text_output = get_qwen_annot(
        messages_list=messages_list
    )
    return text_output

def get_annot_batch(path_img_list, texts_annot_list, idx_annot):
    global model_qwen, processor_qwen
    if model_qwen is None:
        model_qwen, processor_qwen = load_qwen()
    prompts = init_prompts()
    prompt = prompts[idx_annot]
    text_output = get_annot_img(
        path_img_list=path_img_list, 
        texts_annot_list=texts_annot_list,
        prompt=prompt,
    )
    return text_output
