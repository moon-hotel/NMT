"""
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import sys

sys.path.append('../')

from utils import logger_init
from utils import LoadEnglishGermanDataset
import logging

if __name__ == '__main__':
    logger_init(log_file_name='log', log_level=logging.DEBUG, log_dir='./log')
    data_loader = LoadEnglishGermanDataset(batch_size=2, min_freq=2,src_inverse=False)
    train_iter, valid_iter = data_loader.load_train_val_test_data(is_train=True)
    for x, y in train_iter:
        logging.info(x)
        logging.info(y)
        break
