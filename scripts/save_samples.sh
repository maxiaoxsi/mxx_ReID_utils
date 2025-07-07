# !
PATH_MARS="/machangxiao/code/MIP-ReID/configs/datasets/MARS-v160809/cfg_cache_mars_train.yaml"
PATH_MARKET="/machangxiao/code/MIP-ReID/configs/datasets/Market-1501-v15.09.15/cfg_cache_market_train.yaml"

python utils/save_samples.py \
    --stage 1 \
    --is_norm True \
    --is_shuffle True \
    --n_person 10 \
    --path_cfg $PATH_MARKET
    

