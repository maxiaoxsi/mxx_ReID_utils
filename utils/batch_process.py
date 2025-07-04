from mxx.utils.batch import set_annot_smplx_batch

if __name__ == '__main__':
    path_cfg = '/Users/curarpikt/code/mxx/configs/dataset/Market-1501-v15.09.15/cfg_cache_market_train.yaml'
    path_cfg = '/Users/curarpikt/code/mxx/configs/dataset/MSMT17/cfg_cache_msmt17_train.yaml'
    set_annot_smplx_batch(path_cfg=path_cfg)

