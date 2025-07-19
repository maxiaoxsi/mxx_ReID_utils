def load_id_person(name_person, dir_sub):
    return name_person.split('_')[0]

def load_id_video(name_person):
    return None

def load_idx_frame(name_person):
    return None

def get_id_person_min():
    return 1

def get_type_dataset():
    return 'img'

def get_key(basename, dir_sub):
    dir_occ = dir_sub.split('/')[-2]
    return f"{dir_occ}_basename"

