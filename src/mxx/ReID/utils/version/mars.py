def load_id_person(name_person, sub_dir):
    return name_person[:4]

def load_id_video(name_person):
    return name_person[4:11]

def load_id_frame(name_person):
    return name_person[12:15]

def get_id_person_min():
    return 1

def get_type_dataset():
    return 'vid'

def get_key(basename, dir_sub):
    return basename
