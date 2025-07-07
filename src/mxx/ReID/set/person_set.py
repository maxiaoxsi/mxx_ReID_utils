import os
from .set_base import SetBase
from ..object.person import Person
from ..object.person_vid import PersonVid
from ..object.cache import Cache


class PersonSet(SetBase):
    def __init__(self, dataset, logger) -> None:
        super().__init__()
        self._dataset = dataset
        self._logger = logger
        self._keys = []

    def load_cache(
        self, 
        cache:Cache,
    ):
        for id_person, cache_person in cache().items():
            self._keys.append(id_person)
            if self._dataset.type == "img":
                person = Person(
                    id=id_person, 
                    cache_person=cache_person,
                    dataset=self._dataset,
                    logger=self._logger,
                )
            elif self._dataset.type == "vid":
                person = PersonVid(
                    id=id_person,
                    cache_person=cache_person,
                    dataset=self._dataset,
                    logger=self._logger,
                )
            else:
                raise Exception("type dataset key wrong!") 
            self.add_item(id_person, person)

    @property
    def keys(self):
        return self._keys
    
    def __len__(self):
        return len(self._keys)
    
   
        
