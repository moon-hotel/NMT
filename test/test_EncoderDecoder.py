import sys
import torch

sys.path.append('../')

from model import Encoder
from model import DecoderWrapper


def test_Encoder():
    src_input = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9],
                                  [1, 2, 3, 3, 3, 4, 2, 1, 1]])
    embedding_size = 32
    hidden_size = 64
    num_layers = 2
    vocab_size = 100
    cell_type = 'LSTM'
    batch_first = True
    encoder = Encoder(embedding_size, hidden_size, num_layers, vocab_size,
                      cell_type, batch_first)
    output, final_state = encoder(src_input)
    print("encoder output.shape", output.shape)  # [batch_size, src_len, embedding_size]
    print("encoder final_state[0].shape", final_state[0].shape)


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
    batch_first = True
    encoder = Encoder(embedding_size, hidden_size, num_layers, vocab_size,
                      cell_type, batch_first)
    output, final_state = encoder(src_input)

    decoder = DecoderWrapper(embedding_size, hidden_size, num_layers, vocab_size,
                             cell_type, decoder_type='standard', batch_first=True)
    output, final_state = decoder(tgt_input, final_state)
    print("decoder output.shape: ", output.shape)


if __name__ == '__main__':
    test_Encoder()
    test_Decoder()
