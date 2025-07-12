#！/bin/bash
PATH_CFG_MARKET=/machangxiao/code/MIP-ReID/configs/datasets/Market-1501-v15.09.15/cfg_cache_market_train.yaml
PATH_CFG_MARS=/machangxiao/code/MIP-ReID/configs/datasets/MARS-v160809/cfg_cache_mars_train.yaml
PATH_CFG_DUKE=/machangxiao/code/MIP-ReID/configs/datasets/DukeMTMC-reID/cfg_cache_duke_train.yaml
PATH_CFG_MSMT=/machangxiao/code/MIP-ReID/configs/datasets/MSMT17/cfg_cache_msmt17_train.yaml

python run/check_annot_reid.py \
    --path_cfg $PATH_CFG_MARS \
    --batch_size 32 \
    --max_workers 8 \
    --is_riding False \
    --is_backpack False \
    --is_shoulder_bag False \
    --is_hand_carried True \
    --color_upper False \
    --color_bottoms False