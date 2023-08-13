import sys
import torch

sys.path.append('../')

from model import Seq2Seq


class ModelConfig():
    def __init__(self):
        self.src_emb_size = 32
        self.tgt_emb_size = 64
        self.hidden_size = 128
        self.num_layers = 2
        self.src_v_size = 50
        self.tgt_v_size = 60
        self.cell_type = 'GRU'
        self.batch_first = True
        self.dropout = 0.5
        self.decoder_type = 'standard'  # 'luong' or 'standard'


def test_Seq2Seq():
    src_input = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9],
                                  [1, 2, 3, 3, 3, 4, 2, 1, 1]])
    tgt_input = torch.LongTensor([[1, 2, 6, 7, 8, 9],
                                  [1, 2, 4, 2, 1, 1]])
    config = ModelConfig()
    seq2seq = Seq2Seq(config)
    output = seq2seq(src_input, tgt_input)  # [batch_size, tgt_len, hidden_size]
    print("Seq2Seq output.shape: ", output.shape)


if __name__ == '__main__':
    test_Seq2Seq()
