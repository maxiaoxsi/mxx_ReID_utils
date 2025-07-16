#！/bin/bash
PATH_CFG_MARKET=/machangxiao/code/MIP-ReID/configs/datasets/Market-1501-v15.09.15/cfg_cache_market_train.yaml
PATH_CFG_MARS=/machangxiao/code/MIP-ReID/configs/datasets/MARS-v160809/cfg_cache_mars_train.yaml
PATH_CFG_DUKE=/machangxiao/code/MIP-ReID/configs/datasets/DukeMTMC-reID/cfg_cache_duke_train.yaml
PATH_CFG_MSMT=/machangxiao/code/MIP-ReID/configs/datasets/MSMT17/cfg_cache_msmt17_train.yaml

DIR_SUB=v_beta/280000_250713

python run/eval_fid.py \
  --path_cfg $PATH_CFG_MARKET \
  --dir_sub $DIR_SUB