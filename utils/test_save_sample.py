from tqdm import tqdm
from mxx import ReIDProcessor, ReIDDataset
import os


if __name__ == '__main__':
    path_cfg = './configs/dataset/Market-1501-v15.09.15/cfg_cache_market_train.yaml'
   
    dataset_market = ReIDDataset(
        path_cfg=path_cfg,
        img_size=(512, 512),
        stage=1,
        is_select_bernl=True
    )
    from mxx.ReID.utils import save_sample

    for i in range(10):
        sample = dataset_market[i]
        save_sample(
            sample=sample, 
            dir_base=os.path.join('./test', f"img_{i}"),
            is_norm=True,
        )
        print(f'img_{i} sample saved')