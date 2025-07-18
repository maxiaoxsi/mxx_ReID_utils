def load_id_person(name_person, dir_sub):
    return dir_sub.split('/')[-1]

def load_id_video(name_person):
    return None

def load_id_frame(name_person):
    return None

def get_id_person_min():
    return 1

def get_type_dataset():
    return 'img'

def get_key(basename, dir_sub):
    id_cam = dir_sub.split('/')[-2]
    return f'{id_cam}_{basename}'

