import os
import pickle

from mxx.ReID.utils.path import get_path
from ..utils import get_utils
from ...ReID.utils.path import get_basename, get_dir_sub

def add_vid(person, dir_sub, id_vid, id_frame):
    if id_vid in person:
        video = person[id_vid]
    else:
        video = {
            'dir_sub': dir_sub,
            'n_frame': int(id_frame),
        }
        person[id_vid] = video
    if int(id_frame) > video['n_frame']:
        video['n_frame'] = int(id_frame)


def add_person_vid(person_dict, id_person, dir_sub, id_vid, id_frame):
    if id_person in person_dict:
        person = person_dict[id_person]
    else:
        print(f"load person {id_person}")
        person = {}
        person_dict[id_person] = person
    add_vid(person, dir_sub, id_vid, id_frame)

def add_person_img(person_dict, id_person, dir_sub, basename):
    if id_person in person_dict:
        person = person_dict[id_person]
    else:
        print(f"load person {id_person}")
        person = {}
        person_dict[id_person] = person
    
    person[basename] = {
        'dir_sub':dir_sub,
    }

class Cache:
    def __init__(
        self, 
        cfg, 
        logger,
        is_save=True, 
    ):
        self._logger = logger
        self._id_dataset = cfg["id_dataset"]
        self._dir = cfg['dir']
        path_cache = cfg["path_cache"]
        self._cache = {}

        if path_cache is None or not os.path.exists(path_cache):
            self._create_cache()
            if path_cache is not None and is_save:
                dir_cache = os.path.dirname(path_cache)
                os.makedirs(dir_cache, exist_ok=True)
                with open(path_cache, 'wb') as f:
                    pickle.dump(self._cache, f)
        else:
            self._load_cache(path_cache) 
            

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
            self._check_cache_vid()


    def _create_cache_img(self, parser):
        person_dict = {}
        id_person_min = parser.get_id_person_min()
        for root, dirs, files in os.walk(self._dir["reid"]):
            dir_sub = get_dir_sub(root, self._dir)
            for file in files:
                if not file.endswith(('.jpg', '.png')):
                    continue
                basename, ext = get_basename(name_file=file)
                id_person = parser.load_id_person(basename, dir_sub)
                if not id_person.isdigit() or int(id_person) < id_person_min:
                    continue
                add_person_img(person_dict, id_person, dir_sub, basename)
        self._cache['type'] = 'img'
        self._cache['ext'] = ext
        self._cache['person'] = person_dict 

    def _create_cache_vid(self, parser):
        person_dict = {}
        id_person_min = parser.get_id_person_min()
        for root, dirs, files in os.walk(self._dir["reid"]):
            dir_sub = get_dir_sub(root, self._dir)
            for file in files:
                if not file.endswith(('.jpg', '.png')):
                    continue
                basename, ext = get_basename(name_file=file)
                id_person = parser.load_id_person(basename, dir_sub)
                if not id_person.isdigit() or int(id_person) < id_person_min:
                    continue
                id_vid = parser.load_id_video(basename)
                id_frame = parser.load_id_frame(basename) 
                add_person_vid(person_dict, id_person, dir_sub, id_vid, id_frame)
        self._cache['type'] = 'vid'
        self._cache['ext'] = ext
        self._cache['person'] = person_dict 

    def _check_cache_vid(self):
        ext = self._cache['ext']
        for id_p, person in self._cache['person'].items():
            for id_v, vid in person.items():
                n_frame = vid['n_frame']
                for i in range(1, n_frame + 1):
                    basename = f"{id_p}{id_v}F{str(i).zfill(3)}"
                    path_reid = get_path(self._dir, id_p, basename, ext, "reid")
                    if not os.path.exists(path_reid):
                        print(path_reid)
                        print(vid['n_frame'])
                        print(i - 1)
                        vid['n_frame'] = i - 1
                        break

    @property
    def type(self):
        return self._cache['type']

    @property
    def ext(self):
        return self._cache['ext']

    def __call__(self):
        return self._cache['person']


        