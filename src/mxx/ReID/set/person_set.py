import os
from tqdm import tqdm

from PIL import Image
import importlib

from .set_base import SetBase
from ..object import Img
from ..object import Person
from ..object.cache import Cache

class PersonSet(SetBase):
    def __init__(self, dataset, logger) -> None:
        super().__init__()
        self._dataset = dataset
        self._logger = logger
        self._keys = []

    def load_cache(
        self, 
        cache:Cache
    ):
        list_person = cache.get_list_person()
        for person_dict in tqdm(list_person):
            id_person = person_dict['id_person']
            self._keys.append(id_person)
            person = Person(
                id=id_person, 
                cache_img=person_dict['list_img'],
                logger=self._logger,
                dataset=self._dataset,
            )
            self.add_item(id_person, person)

    
    @property
    def keys(self):
        return self._keys
        
