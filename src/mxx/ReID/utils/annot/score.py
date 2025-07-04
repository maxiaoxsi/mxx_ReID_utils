
def add_img_by_score(img_list, img):
    for (i, img_i) in enumerate(img_list):
        if img.score >= img_i.score:
            img_list.insert(i, img)
            return 
    img_list.append(img)
    return