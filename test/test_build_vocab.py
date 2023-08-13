"""
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import sys
import os

sys.path.append('../')

from utils.data_helper import Vocab
from utils.data_helper import my_tokenizer
from utils import DATA_HOME
from utils import logger_init
import logging

if __name__ == '__main__':
    logger_init(log_file_name='log', log_level=logging.INFO, log_dir='./log')
    path_de = os.path.join(DATA_HOME, 'GermanEnglish', 'train_.de')
    tokenizer = my_tokenizer()
    vocab = Vocab(tokenizer['src'], file_path=path_de, min_freq=2, top_k=None)
    logging.info(vocab.stoi)
    logging.info(vocab.itos)
    logging.info(len(vocab))

