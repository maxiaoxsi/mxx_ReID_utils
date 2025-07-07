import argparse
from tqdm import tqdm
from mxx import ReIDDataset
from mxx.ReID.utils import save_sample
import os
import random

def save_samples(dataset, idx):
    print(idx_selected)
    for i in idx_selected:
        sample = dataset[i]
        save_sample(
            sample=sample, 
            dir_base=os.path.join('./test', f"img_{i}"),
            is_norm=True,
        )
        print(f'img_{i} sample saved')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_cfg", type=str, default="/machangxiao/code/MIP-ReID/configs/datasets/MARS-v160809/cfg_cache_mars_train.yaml")
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--is_norm", type=bool, default=True)
    parser.add_argument("--is_shuffle", type=bool, default=False)
    parser.add_argument("--n_person", type=int, default=10)
    
    args = parser.parse_args()
    path_cfg = args.path_cfg

    dataset = ReIDDataset(
        path_cfg=path_cfg,
        img_size=(512, 512),
        stage=1,
        is_select_bernl=True
    )
    if args.is_shuffle:
        idx_selected = random.sample(range(len(dataset)), args.n_person)
    else:
        idx_selected = range(args.n_person)
    save_samples(dataset=dataset, idx=idx_selected)
        