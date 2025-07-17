#！/bin/bash
PATH_CFG_MARKET=/machangxiao/code/MIP-ReID/configs/datasets/Market-1501-v15.09.15/cfg_cache_market_train.yaml
PATH_CFG_MARS=/machangxiao/code/MIP-ReID/configs/datasets/MARS-v160809/cfg_cache_mars_train.yaml
PATH_CFG_DUKE=/machangxiao/code/MIP-ReID/configs/datasets/DukeMTMC-reID/cfg_cache_duke_train.yaml
PATH_CFG_MSMT=/machangxiao/code/MIP-ReID/configs/datasets/MSMT17/cfg_cache_msmt17_train.yaml
PATH_CFG_SYSU=/root/autodl-tmp/code/MIP-ReID/configs/datasets/SYSU-MM01/cfg_cache_sysu_train.yaml

PATH_MARKET_TEST=/machangxiao/code/MIP-ReID/configs/datasets/Market-1501-v15.09.15/cfg_cache_market_test.yaml


python run/annot_dataset_reid.py \
    --path_cfg $PATH_CFG_SYSU \
    --batch_size 256 \
    --max_workers 8 \
    --is_width False \
    --is_smplx False \
    --drn False \
    --is_visible False \
    --is_riding_vl True \
    --is_backpack_vl True \
    --is_shoulder_bag_vl True \
    --is_hand_carried_vl True \
    --upper_vl True \
    --color_upper_vl True \
    --style_upper_vl False \
    --bottoms_vl True \
    --color_bottoms_vl True \
    --style_bottoms_vl False 