import argparse
import os

from mxx.ReID import ReIDDataset
from mxx.ReID.utils.path import load_cfg
from mxx.ReID.utils import save_item

def save_items_all(dataset, cfg):
    dir_ref = cfg["dir"]["ref"]
    for key in dataset.keys:
        save_item(
            dataset=dataset,
            id_person=key,
            idx_img=-1,
            idx_vid=-1,
            dir_base=os.path.join(dir_ref, key),
            is_select_bernl=True
        )
        print(f"img ref {key} saved!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_cfg", type=str, default="/machangxiao/code/MIP-ReID/configs/datasets/MARS-v160809/cfg_cache_mars_train.yaml")
    args = parser.parse_args()
    path_cfg = args.path_cfg
    dataset = ReIDDataset(
        path_cfg=path_cfg,
        img_size=(512, 512),
        stage=1,
        is_select_bernl=True
    )
    cfg = load_cfg(path_cfg=path_cfg)
    save_items_all(dataset, cfg)