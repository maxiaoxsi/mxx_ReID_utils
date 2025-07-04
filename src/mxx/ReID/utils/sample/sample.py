import random

def select_img_bernl(img_list, is_select_bernl):
    if len(img_list) == 0:
        return None
    img_selected = None
    if is_select_bernl:
        for img in img_list:
            if random.random() < 0.5:
                img_selected = img
                break
    if img_selected is None:
        img_selected = img_list[0]
    return img_selected
