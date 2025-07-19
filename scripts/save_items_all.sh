PATH_CFG_MARKET=/machangxiao/code/MIP-ReID/configs/datasets/Market-1501-v15.09.15/cfg_cache_market_train.yaml
PATH_CFG_MARS=/machangxiao/code/MIP-ReID/configs/datasets/MARS-v160809/cfg_cache_mars_train.yaml
PATH_CFG_DUKE=/machangxiao/code/MIP-ReID/configs/datasets/DukeMTMC-reID/cfg_cache_duke_train.yaml
PATH_CFG_MSMT=/machangxiao/code/MIP-ReID/configs/datasets/MSMT17/cfg_cache_msmt17_train.yaml
PATH_CFG_SYSU=/machangxiao/code/MIP-ReID/configs/datasets/SYSU-MM01/cfg_cache_sysu.yaml

python run/save_items_all.py \
    --path_cfg $PATH_CFG_SYSU 