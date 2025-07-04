from tqdm import tqdm
from mxx import ReIDProcessor, ReIDDataset


if __name__ == '__main__':
    path_cfg = './configs/dataset/Market-1501-v15.09.15/cfg_cache_market_train.yaml'
   
    dataset_market = ReIDDataset(
        path_cfg=path_cfg,
        img_size_pad=(512, 512),
        # n_img = (0, 100000),
        stage=1        
    )
    
    processor = ReIDProcessor(
        dataset=dataset_market,
    )
    processor.test_sample('./test', True)