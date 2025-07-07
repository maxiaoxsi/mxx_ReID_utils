from .person import Person
from .video import Video

from ..set.vid_set import VidSet
from ..set.img_set import ImgSet

class PersonVid(Person):
    def __init__(self, id, cache_person, dataset, logger)->None:
        self._video_set = None
        super().__init__(id, cache_person, dataset, logger)

    def _load_cache(self):
        self._vid_set = VidSet()
        self._img_set = ImgSet()
        for id_vid, cache_vid in enumerate(self._cache_person):
            vid = Video(
                id_vid=id_vid,
                cache_vid=cache_vid,
                dataset=self._dataset,
                logger=self._logger,
            )
            self._vid_set.add_item(id_vid, vid)
            # img_ref_list = vid.get_img_ref_list
            # for img in img_ref_list:
            #     pass
        del self._cache_person
