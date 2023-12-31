"""
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import sys
import torch

sys.path.append('../')
from model import Encoder
from model import DecoderWrapper
from model import LuongAttention
import matplotlib.pyplot as plt

src_input = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9],
                              [1, 2, 3, 3, 3, 4, 2, 1, 1]])
src_key_padding_mask = torch.tensor([[False, False, False, False, False, False, True, True, True],
                                     [False, False, False, False, True, True, True, True, True]])
tgt_input = torch.LongTensor([[1, 2, 6, 7, 8, 9],
                              [1, 2, 4, 2, 1, 1]])
embedding_size = 32
hidden_size = 64
num_layers = 2
vocab_size = 100
cell_type = 'GRU'
batch_first = True
batch_size, src_len = src_input.shape


def test_LuongAttentionDecoder():
    encoder = Encoder(embedding_size, hidden_size, num_layers, vocab_size,
                      cell_type, batch_first)
    output, final_state = encoder(src_input)
    decoder = DecoderWrapper(embedding_size, hidden_size, num_layers, vocab_size,
                             cell_type, attention_type='luong', batch_first=True)
    output, final_state = decoder(tgt_input, final_state, output, src_key_padding_mask)
    print("decoder output.shape: ", output.shape)


def test_attention():
    luong = LuongAttention(hidden_size)
    query = torch.rand((batch_size, hidden_size))
    key = value = torch.rand((batch_size, src_len, hidden_size))
    context_vec, attention_weights = luong(query, key, value, src_key_padding_mask)
    print(context_vec)
    print(context_vec.shape)  # [batch_size, 1, hidden_size]
    print(attention_weights.shape)  # [batch_size, src_len]

    plt.imshow(attention_weights.detach(), cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title('Attention Matrix (without mask)')
    plt.xlabel('Time Steps')
    plt.ylabel('Batch Size')
    plt.show()


if __name__ == '__main__':
    test_LuongAttentionDecoder()
    test_attention()
