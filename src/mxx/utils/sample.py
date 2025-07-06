from mxx.ReID.utils import save_sample, save_item
from tqdm import tqdm
import os

def test_save_sample(samples, dir_save, is_norm=True):
    for i, sample in enumerate(samples):
        save_sample(
            sample=sample,
            dir_base=os.path.join(dir_save, f"img_{i}"),
            is_norm=is_norm,
        )
        print(f'img_{i} sample saved')

def test_save_item(dataset, dir_save, n_sample=10):
    for i in tqdm(range(n_sample)):
        save_item(
            dataset=dataset,
            id_person=i,
            idx_video_tgt=-1,
            idx_img_tgt=-1,
            dir_base=os.path.join(dir_save, f"person_{i}"),
            is_select_bernl=False,
        )
        print(f'item_{i} saved')