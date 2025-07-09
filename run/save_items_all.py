import argparse
from mxx.utils.batch import process_batch

def make_mask(path_cfg):
    from mxx.utils.mask import make_mask_img
    process_batch(
        path_cfg=path_cfg,
        name_proc="mask dataset",
        method=make_mask_img,
        batch_size=256,
        max_workers=8,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_cfg", type=str, default="/machangxiao/code/MIP-ReID/configs/datasets/MARS-v160809/cfg_cache_mars_train.yaml")
    args = parser.parse_args()
    path_cfg = args.path_cfg
    make_mask(path_cfg)
