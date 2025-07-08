from ..set.vid_set import VidSet
from ..set.img_set import ImgSet

from .person import Person
from .video import Video

import random

class PersonVid(Person):
    def __init__(self, id, cache, dataset, logger)->None:
        super().__init__(id, cache, dataset, logger)
        self._vid_set = None

    def _load_cache(self):
        self._vid_set = VidSet()
        self._img_set = ImgSet()
        for id_vid, cache_vid in self._cache.items():
            vid = Video(
                id=id_vid,
                cache=cache_vid,
                dataset=self._dataset,
                person=self,
                logger=self._logger,
            )
            self._vid_set.add_item(id_vid, vid)
            img_ref_list = vid.img_ref_list
            for img in img_ref_list:
                self._img_set.add_item(img.basename, img)
        del self._cache

    def get_img_tgt_list(self, idx_vid, idx_img, n_frame, stage):
        if idx_vid == -1:
            idx_vid = random.randint(0, len(self._vid_set) - 1)
        idx_vid = idx_vid % len(self._vid_set)
        return self._vid_set[idx_vid].get_img_tgt_list(n_frame, stage)

