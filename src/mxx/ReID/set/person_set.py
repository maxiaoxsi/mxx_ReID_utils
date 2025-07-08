import os
from .set_base import SetBase
from ..object.person import Person
from ..object.person_vid import PersonVid
from ..object.cache import Cache


class PersonSet(SetBase):
    def __init__(self, dataset, logger, cache=None) -> None:
        super().__init__()
        self._dataset = dataset
        self._logger = logger
        self._keys = []
        if cache is not None:
            self.load_cache(cache)

    def load_cache(
        self, 
        cache:Cache,
    ):
        if self.type == "img":
            PersonCache = Person
        elif self.type == "vid":
            PersonCache = PersonVid
        else:
            raise Exception("type dataset key wrong!") 
        for id_person, cache_person in cache().items():
            self._keys.append(id_person)
            person = PersonCache(
                id=id_person, 
                cache=cache_person,
                dataset=self._dataset,
                logger=self._logger,
            )
            self.add_item(id_person, person)

    @property
    def keys(self):
        return self._keys
    
    @property
    def type(self):
        return self._dataset.type

    def __len__(self):
        return len(self._keys)
    
   
        
