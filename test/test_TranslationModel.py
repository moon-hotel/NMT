import sys
import torch

sys.path.append('../')

from model import TranslationModel


def test_TranslationModel():
    src_input = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9],
                                  [1, 2, 3, 3, 3, 4, 2, 1, 1]])
    tgt_input = torch.LongTensor([[1, 2, 6, 7, 8, 9],
                                  [1, 2, 4, 2, 1, 1]])
    src_emb_size = 32
    tgt_emb_size = 64
    hidden_size = 128
    num_layers = 2
    src_v_size = 50
    tgt_v_size = 60
    cell_type = 'LSTM'
    bidirectional = False
    batch_first = True
    model = TranslationModel(src_emb_size, tgt_emb_size, hidden_size, num_layers, src_v_size, tgt_v_size,
                             cell_type, bidirectional, batch_first)
    logits = model(src_input, tgt_input)
    print(logits.shape)


if __name__ == '__main__':
    test_TranslationModel()
