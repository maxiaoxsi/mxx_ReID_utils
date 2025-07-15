import os
import pickle

from mxx.ReID.utils.path import get_path
from ..utils import get_utils
from ...ReID.utils.path import get_basename, get_dir_sub

def add_vid(person, dir_sub, id_vid, id_frame, is_smplx):
    if id_vid in person:
        video = person[id_vid]
    else:
        video = {
            'dir_sub': dir_sub,
            'n_frame': int(id_frame),
            'frame_with_smplx': [],
            'frame_without_smplx': [],
        }
        person[id_vid] = video
    if int(id_frame) > video['n_frame']:
        video['n_frame'] = int(id_frame)
    if is_smplx:
        video['frame_with_smplx'].append(int(id_frame))
    else:
        video['frame_without_smplx'].append(int(id_frame))

def add_person_vid(person_dict, id_person, dir_sub, id_vid, id_frame, is_smplx):
    if id_person in person_dict:
        person = person_dict[id_person]
    else:
        print(f"load person {id_person}")
        person = {}
        person_dict[id_person] = person
    add_vid(person, dir_sub, id_vid, id_frame, is_smplx)

def add_person_img(person_dict, id_person, dir_sub, basename, is_smplx):
    if id_person in person_dict:
        person = person_dict[id_person]
    else:
        print(f"load person {id_person}")
        person = {}
        person_dict[id_person] = person
    
    person[basename] = {
        'dir_sub':dir_sub,
        'is_smplx':is_smplx,
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
            self._check_cache_img()
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
                path_annot = get_path(self._dir, dir_sub, basename, ext, "annot")
                path_manikin = get_path(self._dir, dir_sub, basename, ext, "manikin")
                from ...annot.annot_base import AnnotBase
                annot = AnnotBase(path_annot=path_annot, logger=self._logger)
                if 'is_smplx' in annot:
                    is_smplx = annot.get_annot('is_smplx')
                    if is_smplx in ['True', True]:
                        is_smplx = True
                    else:
                        is_smplx = False
                else:
                    is_smplx = os.path.exists(path_manikin)
                    annot.set_annot('is_smplx', is_smplx)
                add_person_img(person_dict, id_person, dir_sub, basename, is_smplx)
        self._cache['type'] = 'img'
        self._cache['ext'] = ext
        self._cache['person'] = person_dict 

    def _check_cache_img(self):
        person_dict = {}
        for id_p, person in self._cache['person'].items():
            for basename, img in person.items():
                if img['is_smplx'] in ['True', True]:
                    person_dict[id_p] = person
                    break
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
                path_annot = get_path(self._dir, dir_sub, basename, ext, "annot")
                from ...annot.annot_base import AnnotBase
                annot = AnnotBase(path_annot=path_annot, logger=self._logger)
                is_smplx = annot.get_annot('is_smplx')
                if is_smplx in ['True', True]:
                    is_smplx = True
                elif is_smplx in ['False', False]:
                    is_smplx = False
                else:
                    self._logger(f'[cache] is_smplx key wrong!')
                add_person_vid(person_dict, id_person, dir_sub, id_vid, id_frame, is_smplx)
        self._cache['type'] = 'vid'
        self._cache['ext'] = ext
        self._cache['person'] = person_dict 

    def _check_cache_vid(self):
        ext = self._cache['ext']
        cache = {}
        for id_p, person in self._cache['person'].items():
            person_new = {}
            for id_v, vid in person.items():
                n_frame = vid['n_frame']
                for i in range(1, n_frame + 1):
                    basename = f"{id_p}{id_v}F{str(i).zfill(3)}"
                    path_reid = get_path(self._dir, id_p, basename, ext, "reid")
                    if not os.path.exists(path_reid):
                        print(path_reid)
                        print(vid['n_frame'])
                        print(i)
                        vid['n_frame'] = i - 1
                        vid['frame_with_smplx'] = [item for item in vid['frame_with_smplx'] if item < i]
                        vid['frame_without_smplx'] = [item for item in vid['frame_without_smplx'] if item < i]
                        break
                if len(vid['frame_with_smplx']) > 0:
                    person_new[id_v] = vid
                else:
                    print(f"{id_v} all frame without smplx")
            if len(person_new) > 0:
                cache[id_p] = person_new
        self._cache['person'] = cache

    @property
    def type(self):
        return self._cache['type']

    @property
    def ext(self):
        return self._cache['ext']

    def __call__(self):
        return self._cache['person']


        