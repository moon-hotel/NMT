import sys
import torch

sys.path.append('../')

from model import Encoder
from model import Decoder
from model import Seq2Seq


def test_Encoder():
    src_input = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9],
                                  [1, 2, 3, 3, 3, 4, 2, 1, 1]])
    embedding_size = 32
    hidden_size = 64
    num_layers = 2
    vocab_size = 100
    cell_type = 'LSTM'
    bidirectional = False
    batch_first = True
    encoder = Encoder(embedding_size, hidden_size, num_layers, vocab_size,
                      cell_type, bidirectional, batch_first)
    output, final_state = encoder(src_input)
    print(output.shape)  # [batch_size, src_len, embedding_size]
    print(final_state[0].shape)


def test_Decoder():
    src_input = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9],
                                  [1, 2, 3, 3, 3, 4, 2, 1, 1]])
    tgt_input = torch.LongTensor([[1, 2, 6, 7, 8, 9],
                                  [1, 2, 4, 2, 1, 1]])
    embedding_size = 32
    hidden_size = 64
    num_layers = 2
    vocab_size = 100
    cell_type = 'LSTM'
    bidirectional = False
    batch_first = True
    encoder = Encoder(embedding_size, hidden_size, num_layers, vocab_size,
                      cell_type, bidirectional, batch_first)
    output, final_state = encoder(src_input)

    decoder = Decoder(embedding_size, hidden_size, num_layers, vocab_size,
                      cell_type, bidirectional, batch_first)
    output, final_state = decoder(tgt_input, final_state)
    print(output.shape)


def test_Seq2Seq():
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
    seq2seq = Seq2Seq(src_emb_size, tgt_emb_size, hidden_size, num_layers, src_v_size, tgt_v_size,
                      cell_type, bidirectional, batch_first)
    output = seq2seq(src_input, tgt_input) # [batch_size, tgt_len, hidden_size]
    print(output.shape)


if __name__ == '__main__':
    # test_Encoder()
    # test_Decoder()
    test_Seq2Seq()
