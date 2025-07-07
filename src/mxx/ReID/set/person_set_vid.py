from .person_set import PersonSet
from ..object.cache import Cache
from .video_set import VideoSet
from .img_set import ImgSet

class VidPersonSet(PersonSet):
    def __init__(self, dataset, logger) -> None:
        super().__init__(dataset, logger)
    
    def load_cache(self, cache: Cache):
        self._vid_set = VideoSet()
        self._img_set = ImgSet()
        