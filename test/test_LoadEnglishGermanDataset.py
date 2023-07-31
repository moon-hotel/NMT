import sys
import os

sys.path.append('../')

from utils import DATA_HOME
from utils import logger_init
from utils import LoadEnglishGermanDataset
import logging

if __name__ == '__main__':
    logger_init(log_file_name='log', log_level=logging.INFO, log_dir='./log')
    path_de = os.path.join(DATA_HOME, 'GermanEnglish', 'train_.de')
    path_en = os.path.join(DATA_HOME, 'GermanEnglish', 'train_.en')
    path = {'src': path_de, 'tgt': path_en}
    data_loader = LoadEnglishGermanDataset(train_file_paths=path, batch_size=2, min_freq=2, )
    train_iter, valid_iter, test_iter = data_loader.load_train_val_test_data(path,path,path)
    for x, y in train_iter:
        logging.info(x)
        logging.info(y)
        break
