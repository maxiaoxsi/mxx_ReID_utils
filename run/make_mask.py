import argparse
from mxx.utils.batch import process_reid_batch

def make_mask(path_cfg, batch_size, max_workers):
    from mxx.utils.mask import make_mask_img
    process_reid_batch(
        path_cfg=path_cfg,
        name_processing="mask dataset",
        method=make_mask_img,
        batch_size=batch_size,
        max_workers=max_workers,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_cfg", type=str, default="/machangxiao/code/MIP-ReID/configs/datasets/MARS-v160809/cfg_cache_mars_train.yaml")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_workers", type=int, default=8)
    
    
    args = parser.parse_args()
    path_cfg = args.path_cfg
    batch_size = args.batch_size
    max_workers = args.max_workers
    make_mask(path_cfg, batch_size, max_workers)
