from .img import Img
import random


class Video:
    def __init__(self, id, cache, dataset, person, logger) -> None:
        self._id = id
        self._dataset = dataset
        self._person = person
        self._logger = logger
        self._dir_sub = cache["dir_sub"]
        self._n_frame = cache["n_frame"]
        self._frame_without_smplx_list = cache["frame_without_smplx"]
        self._frame_with_smplx_list = []
        for i in range(1, self.n_frame + 1):
            if i not in self._frame_without_smplx_list:
                self._frame_with_smplx_list.append(i)
        self._img_list = []
        for i in range(self._n_frame):
            self._img_list.append(None)

    def __getitem__(self, idx):
        idx = idx - 1
        idx = idx % (self.n_frame)
        if self._img_list[idx] is None:
            cache = {
                "dir_sub":self.dir_sub,
                "is_smplx":(idx + 1) in self._frame_without_smplx_list,
            }
            img = Img(
                basename=self.get_basename(idx + 1),
                cache=cache,
                dataset = self._dataset,
                person=self._person,
                logger=self._logger,
            )
            self._img_list[idx] = img
        else:
            img = self._img_list[idx]
        return img

    def get_img_tgt_list(self, n_frame, stage):
        if n_frame < 2:
            n_frame = 2
        if n_frame > self.n_frame:
            idx_st = 1
        else:
            idx_st = random.randint(1, (self.n_frame - n_frame + 1))
        img_tgt_list = []
        for i in range(idx_st, idx_st + n_frame):
            img_tgt_list.append(self[i] if i < self.n_frame else self[self.n_frame])
        return img_tgt_list
    
    @property
    def img_ref_list(self):
        img_ref_list = []
        if len(self._frame_with_smplx_list) < 3:
            for idx in self._frame_with_smplx_list:
                img_ref_list.append(self[idx])
            return img_ref_list
        idx_list = []
        idx_list.append(self._frame_with_smplx_list[0])
        idx_list.append(self._frame_with_smplx_list[-1])
        idx_mid = random.randint(1, len(self._frame_with_smplx_list) - 2)
        idx_list.append(self._frame_with_smplx_list[idx_mid])
        img_ref_list = [self[idx] for idx in idx_list]
        return img_ref_list

    @property
    def n_frame(self):
        return self._n_frame
    
    @property
    def dir_sub(self):
        return self._dir_sub
    
    @property
    def id(self):
        return self._id

    def get_basename(self, id_frame):
        return f"{self._person.id}{self.id}F{str(id_frame).zfill(3)}"


    
        