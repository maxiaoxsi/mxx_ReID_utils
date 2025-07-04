import os
import pickle
from ..utils import get_utils


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
        self._path_cache = cfg["path_cache"] + '.pkl'
        self._cache = {}

        if self._path_cache is None or not os.path.exists(self._path_cache):
            self._create_cache()
            if self._path_cache is not None and is_save:
                dir_cache = os.path.dirname(self._path_cache)
                os.makedirs(dir_cache, exist_ok=True)
                with open(self._path_cache, 'wb') as f:
                    pickle.dump(self._cache, f)
        else:
            if not is_divide:
                self._load_cache(self._path_cache) 
            else:
                st_divide = cfg["st_divide"]
                n_divide = cfg["n_divide"]
                if st_divide < 0:
                    st_divide = 0
                if n_divide < 0:
                    n_divide = 0
                for i in range(st_divide, st_divide + n_divide):
                    path_cache_divide = cfg["path_cache"] + f'_{i}.pkl'
                    if os.path.exists(path_cache_divide):
                        self._load_cache(path_cache_divide)
                    else:
                        break
            

    def _load_cache(self, path_cache):
        with open(path_cache, 'rb') as f:
            cache = pickle.load(f)
            if self._cache == {}:
                self._cache = cache
            elif self._cache['type'] == 'img':
                self._cache['list_person'] = self._cache['list_person'] + cache['list_person']

    def _create_cache(self):
        parser = get_utils(id_dataset=self._id_dataset)
        self._cache["type"] = parser.get_type_dataset()
        if self._cache["type"] == 'img':
            self._create_cache_img(parser)
        elif self._cache["type"] == "vid":
            self._create_cache_vid(parser)


    def _create_cache_img(self, parser):
        person_dict = {}
        person_list = []
        id_person_min = parser.get_id_person_min()
        for root, dirs, files in os.walk(self._dir["reid"]):
            dir_sub = root[len(self._dir["reid"]) + 1:]
            for file in files:
                if not file.endswith(('.jpg', '.png')):
                    continue
                name_file = file.split('/')[-1]
                suff_file = name_file.split('.')[-1]
                name_file = name_file.split('.')[0]
                id_person = parser.load_id_person(name_file, dir_sub)
                if not id_person.isdigit() or int(id_person) < id_person_min:
                    continue
                # id_frame = parser.load_id_frame(name_file)
                # id_video = parser.load_id_video(name_file)
                path_smplx_guid = os.path.join(self._dir["smplx"], 
                    dir_sub, 'manikin', f'{name_file}.{suff_file}')
                is_smplx = os.path.exists(path_smplx_guid)
                if not is_smplx:
                    self._logger(f"[cache] img:{name_file} smplx img not found!")
                img_dict = {
                    'dir_sub':dir_sub,
                    'name':name_file,
                    'suff':suff_file,
                    'is_smplx':is_smplx,
                }
                if id_person in person_dict:
                    person_dict[id_person]['list_img'].append(img_dict)
                else:
                    person = {
                        'id_person' : id_person,
                        'list_img': [img_dict],
                    }
                    person_dict[id_person] = person
                    person_list.append(person)
        
        person_list = [person for person in person_list if any(img['is_smplx'] for img in person['list_img'])]
        person_list.sort(key=lambda x: int(x['id_person']))
        self._cache['list_person'] = person_list

    @property
    def person_keys(self):
        return [item['id_person'] for item in self._cache['list_person']]

    def _create_cache_vid(self, parser):
        print("not finish")

    def get_list_person(self):
        return self._cache['list_person']


        