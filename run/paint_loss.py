import argparse
import os
from mxx import Logger

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str)
    parser.add_argument("--name_exp", type=str)
    args = parser.parse_args()
    exp = args.exp
    name_exp = args.name_exp
    logger = Logger(os.path.join(exp, name_exp, 'train.log'))
    logger.paint_loss()
