import sys
import yaml
import os
import pickle

from mxx.ReID.processor import ReIDProcessor
from mxx.ReID.dataset import ReIDDataset

if __name__ == '__main__':
    dir_cache = '/Users/curarpikt/Desktop/mxx/configs/dataset/MSMT17'
    processor = ReIDProcessor()
    processor.sort_and_divide_cache(dir_cache=dir_cache, name_base="cache_msmt17_train", num_divided=3)