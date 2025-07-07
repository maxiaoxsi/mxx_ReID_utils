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
        for id_person, cache_person in enumerate(cache()):
            self._keys.append(id_person)
            person = Person(
                id=id_person, 
                cache_person=cache_person,
                logger=self._logger,
                dataset=self._dataset,
            )
            self.add_item(id_person, person)
    
    @property
    def keys(self):
        return self._keys
        
