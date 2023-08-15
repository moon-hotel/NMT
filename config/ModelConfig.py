"""
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""
import os
import logging
from utils import logger_init
from utils import get_gpus
from utils import PROJECT_HOME


class TranslationModelConfig(object):
    def __init__(self, show_paras=False):
        self.src_emb_size = 128
        self.tgt_emb_size = 128
        self.hidden_size = 256
        self.num_layers = 3
        self.cell_type = 'GRU'
        self.batch_first = True
        self.src_v_size = None  # 等价于 src_top_k
        self.tgt_v_size = None  # 等价于 tgt_top_k
        self.batch_size = 64
        self.min_freq = 2
        self.src_inverse = False  # 是否反转输入序列
        self.devices = get_gpus(1)
        self.epochs = 200
        self.learning_rate = 0.002
        self.attention_type = 'luong'  # 'luong' or 'standard' or 'bahdanau'
        self.dropout = 0.1
        self.num_warmup_steps = 300
        self.model_save_dir = os.path.join(PROJECT_HOME, 'cache')
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        self.model_save_path = os.path.join(self.model_save_dir, 'trans_model.pt')
        logger_init(log_file_name='log', log_level=logging.DEBUG, log_dir='./log')

        if show_paras:
            self.show_paras()

    def show_paras(self):
        logging.info(f"打印模型当前参数")
        for k, v in self.__dict__.items():
            logging.info(f"{k} =  {v}")
