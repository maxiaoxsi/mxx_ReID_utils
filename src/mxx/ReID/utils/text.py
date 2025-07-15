from ..object.img import Img

def get_text_backpack(img):
    if img["is_backpack"]:
        return f", with a backpack"
    return ""

def get_text_hand_carried(img):
    if img['is_hand_carried_vl']:
        return f", with an object in his hand"
    return ""

def get_text_drn(img):
    if img['is_riding']:
        text_walk = 'riding'
    else:
        text_walk = 'walking'
    drn = img['drn']
    if drn == 'left':
        return f', {text_walk} from right to left'
    if drn == 'right':
        return f', {text_walk} from left to right'
    if drn == 'front':
        return f', {text_walk} toward the camera'
    if drn == 'back':
        return f', {text_walk} away from the camera'
    return f', {text_walk} {drn}'