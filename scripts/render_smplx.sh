PATH_MARKET_TRAIN=/machangxiao/code/MIP-ReID/configs/datasets/Market-1501-v15.09.15/cfg_cache_market_train.yaml
PATH_MARS_TRAIN=/machangxiao/code/MIP-ReID/configs/datasets/MARS-v160809/cfg_cache_mars_train.yaml
PATH_DUKE_TRAIN=/machangxiao/code/MIP-ReID/configs/datasets/DukeMTMC-reID/cfg_cache_duke_train.yaml
PATH_MSMT_TRAIN=/machangxiao/code/MIP-ReID/configs/datasets/MSMT17/cfg_cache_msmt17_train.yaml
PATH_MARKET_TEST=/machangxiao/code/MIP-ReID/configs/datasets/Market-1501-v15.09.15/cfg_cache_market_test.yaml
PATH_OCC_REID=/machangxiao/code/MIP-ReID/configs/datasets/Occluded_REID/cfg_cache_occ_reid.yaml


python run/render_smplx.py \
    --path_cfg $PATH_OCC_REID \
    --batch_size 256 \
    --max_workers 4