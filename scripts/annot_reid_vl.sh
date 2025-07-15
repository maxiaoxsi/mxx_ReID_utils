# 设置环境变量（注意等号两边不能有空格）
export PATH_MARS=/machangxiao/code/MIP-ReID/configs/datasets/MARS-v160809/cfg_cache_mars_train.yaml
export PATH_MARKET=/machangxiao/code/MIP-ReID/configs/datasets/Market-1501-v15.09.15/cfg_cache_market_train.yaml
export PATH_MSMT=/machangxiao/code/MIP-ReID/configs/datasets/MSMT17/cfg_cache_msmt17_train.yaml
export PATH_DUKE=/machangxiao/code/MIP-ReID/configs/datasets/DukeMTMC-reID/cfg_cache_duke_train.yaml


python utils/annot_reid_vl.py \
    --path_cfg $PATH_MARS \
    --annot "is_backpack_vl" \
    --batch_size 64