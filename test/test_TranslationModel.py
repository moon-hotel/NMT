"""
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import sys
import torch

sys.path.append('../')

from model import TranslationModel
from model import greedy_decode


class ModelConfig():
    def __init__(self):
        self.src_emb_size = 32
        self.tgt_emb_size = 64
        self.hidden_size = 128
        self.num_layers = 2
        self.src_v_size = 50
        self.tgt_v_size = 60
        self.cell_type = 'LSTM'
        self.batch_first = True
        self.dropout = 0.5
        self.attention_type = 'standard'


def test_TranslationModel():
    src_input = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9],
                                  [1, 2, 3, 3, 3, 4, 2, 1, 1]])
    tgt_input = torch.LongTensor([[1, 2, 6, 7, 8, 9],
                                  [1, 2, 4, 2, 1, 1]])
    config = ModelConfig()
    model = TranslationModel(config)
    logits = model(src_input, tgt_input)
    print(logits.shape)


def test_TranslationModelInference():
    """
    标准解码器下贪婪策略的解码过程
    :return:
    """
    src_input = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    config = ModelConfig()
    model = TranslationModel(config)
    results = greedy_decode(model, src_input, 2, 3, device='cpu')
    print(results)


if __name__ == '__main__':
    test_TranslationModel()
    test_TranslationModelInference()
