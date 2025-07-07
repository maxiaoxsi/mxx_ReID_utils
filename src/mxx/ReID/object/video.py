from .img import Img
import random


class Video:
    def __init__(self, id_vid, cache_vid, dataset, logger) -> None:
        self._id_vid = id_vid
        self._dataset = dataset
        self._logger = logger
        self._dir_sub = cache_vid["dir_sub"]
        self._n_frame = cache_vid["n_frame"]
        self._frame_without_smplx = cache_vid["frame_without_smplx"]
        for i in range(self._n_frame):
            self._img_list.append(None)
        print(self._n_frame)

    def get_img_tgt_list(self, n_frame):
        print("finish later")
        pass

    # def get_img_tgt_list(self, idx_img_tgt, n_frame):
    #     if len(self._img_list) == 0:
    #         self._init_img_list()
    #     if len(self._img_list) == 0:
    #         print("img_list empty")
    #         return []
    #     if n_frame < 0:
    #         n_frame = 10
    #     if idx_img_tgt > 0:
    #         frame_st = idx_img_tgt % len(self._img_list)
    #     else:
    #         if n_frame < len(self._img_list):
    #             frame_st = random.randint(0, len(self._img_list) - n_frame)
    #         else:
    #             frame_st = 0
    #     ans = self._img_list[frame_st: frame_st + n_frame]
    #     for i in range(n_frame - len(ans)):
    #         ans.append(self._img_list[-1])
    #     return ans
        