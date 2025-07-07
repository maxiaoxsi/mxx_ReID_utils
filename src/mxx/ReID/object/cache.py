import os
import pickle

from mxx.ReID.utils.path import get_path_manikin
from ..utils import get_utils
from ...utils.batch import load_name

def add_vid(person, dir_sub, id_vid, id_frame, is_smplx):
    if id_vid not in person:
        person[id_vid] = {
            'dir_sub': dir_sub,
            'frame': int(id_frame),
            'not_smplx': [],
        }
    video = person[id_vid]
    if int(id_frame) > person['frame']:
        person['frame'] = int(id_frame)
    if not is_smplx:
        video['not_smplx'].append(int(id_frame))


def add_person_vid(person_dict, id_person, dir_sub, id_vid, id_frame, is_smplx):
    if id_person not in person_dict:
        person_dict[id_person] = {}
    person = person_dict[id_person]
    add_vid(person, dir_sub, id_vid, id_frame, is_smplx)

def add_person_img(person_dict, id_person, dir_sub, 
        name, suff, is_smplx):
    if id_person in person_dict:
        person = person_dict[id_person]
    else:
        person = {}
        person_dict[id_person] = person
    person[name] = {
        'dir_sub':dir_sub,
        'name':name,
        'suff':suff,
        'is_smplx':is_smplx,
    }

class Cache:
    def __init__(
        self, 
        cfg, 
        logger,
        is_save=True, 
        is_divide=False, 
    ):
        self._logger = logger
        self._id_dataset = cfg["id_dataset"]
        self._dir = cfg['dir']
        self._path_cache = cfg["path_cache"]
        self._cache = {}

        if self._path_cache is None or not os.path.exists(self._path_cache):
            self._create_cache()
            if self._path_cache is not None and is_save:
                dir_cache = os.path.dirname(self._path_cache)
                os.makedirs(dir_cache, exist_ok=True)
                with open(self._path_cache, 'wb') as f:
                    pickle.dump(self._cache, f)
        else:
            self._load_cache(self._path_cache) 
            

    def _load_cache(self, path_cache):
        with open(path_cache, 'rb') as f:
            cache = pickle.load(f)
            self._cache = cache


    def _create_cache(self):
        parser = get_utils(id_dataset=self._id_dataset)
        self._cache["type"] = parser.get_type_dataset()
        if self._cache["type"] == 'img':
            self._create_cache_img(parser)
        elif self._cache["type"] == "vid":
            self._create_cache_vid(parser)


    def _create_cache_vid(self, parser):
        person_dict = {}
        id_person_min = parser.get_id_person_min()
        for root, dirs, files in os.walk(self._dir["reid"]):
            dir_sub = root[len(self._dir["reid"]) + 1:]
            for file in files:
                if not file.endswith(('.jpg', '.png')):
                    continue
                name_file, suff_file = load_name(file=file)
                id_person = parser.load_id_person(name_file, dir_sub)
                if not id_person.isdigit() or int(id_person) < id_person_min:
                    continue
                id_vid = parser.load_id_video(name_file)
                id_frame = parser.load_id_frame(name_file) 
                from ..utils.path import get_path_manikin
                is_smplx = os.path.exists(get_path_manikin(dir, dir_sub, file))
                add_person_vid(person_dict, id_person, dir_sub, id_vid, id_frame, is_smplx)
                self._cache['type'] = 'vid'
                self._cache['person'] = person_dict 

    def _create_cache_img(self, parser):
        person_dict = {}
        id_person_min = parser.get_id_person_min()
        for root, dirs, files in os.walk(self._dir["reid"]):
            dir_sub = root[len(self._dir["reid"]) + 1:]
            for file in files:
                if not file.endswith(('.jpg', '.png')):
                    continue
                id_person = parser.load_id_person(name_file, dir_sub)
                if not id_person.isdigit() or int(id_person) < id_person_min:
                    continue
 
                name_file, suff_file = load_name(file=file)
                path_manikin = get_path_manikin(self._dir['smplx'], dir_sub, file)
                is_smplx = os.path.exists(path_manikin)

                add_person_img(person_dict, id_person, dir_sub, name_file, suff_file, is_smplx)
        self._cache['person'] = person_dict

    @property
    def type(self):
        return self._cache['type']

    def __call__(self):
        return self._cache['person']


        