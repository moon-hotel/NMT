import os
import logging
from utils import logger_init
from utils import get_gpus
from utils import PROJECT_HOME



class TranslationModelConfig():
    def __init__(self):
        self.src_emb_size = 256
        self.tgt_emb_size = 256
        self.hidden_size = 512
        self.num_layers = 2
        self.cell_type = 'LSTM'
        self.bidirectional = False
        self.batch_first = True
        self.src_v_size = None  # 等价于 src_top_k
        self.tgt_v_size = None  # 等价于 tgt_top_k
        self.batch_size = 64
        self.min_freq = 2
        self.src_inverse = False  # 是否反转输入序列
        self.devices = get_gpus(1)
        self.epochs = 10
        self.learning_rate = 0.001
        self.model_save_dir = os.path.join(PROJECT_HOME, 'cache')
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        self.model_save_path = os.path.join(self.model_save_dir, 'trans_model.pt')
        logger_init(log_file_name='log', log_level=logging.DEBUG, log_dir='./log')
