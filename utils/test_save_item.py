from tqdm import tqdm
from mxx import ReIDProcessor, ReIDDataset
import os


if __name__ == '__main__':
    path_cfg = './configs/dataset/Market-1501-v15.09.15/cfg_cache_market_train.yaml'
   
    path_cfg = '/Users/curarpikt/code/mxx/configs/dataset/MARS-v160809/cfg_cache_mars_train.yaml'
    
    dataset_market = ReIDDataset(
        path_cfg=path_cfg,
        img_size=(512, 512),
        stage=1,
        is_select_bernl=True
    )
    from mxx.ReID.utils import save_item

    for i in range(10):
        save_item(
            dataset=dataset_market,
            id_person=i,
            idx_video_tgt=-1,
            idx_img_tgt=-1,
            dir_base=os.path.join('./test_save_item', f"img_{i}"),
            is_select_bernl=False,
        )
        print(f'img_{i} sample saved')