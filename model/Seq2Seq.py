import torch.nn as nn
import torch


class Encoder(nn.Module):
    """
    解码器
    """

    def __init__(self, embedding_size, hidden_size, num_layers, vocab_size,
                 cell_type='LSTM', batch_first=True, dropout=0.):
        """
        :param embedding_size:
        :param hidden_size:
        :param num_layers:  RNN的层数
        :param vocab_size:
        :param cell_type:
        :param batch_first:
        """
        super(Encoder, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.cell_type = cell_type
        self.batch_first = batch_first
        self.dropout = dropout

        if cell_type == 'LSTM':
            rnn_cell = nn.LSTM
        elif cell_type == 'GRU':
            rnn_cell = nn.GRU
        else:
            raise ValueError("Unrecognized RNN cell type: " + cell_type)

        self.token_embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.rnn = rnn_cell(self.embedding_size, self.hidden_size, num_layers=self.num_layers,
                            batch_first=self.batch_first, dropout=self.dropout)

    def forward(self, src_input=None):
        """

        :param src_input: [batch_size, src_len] 这种情况 batch_first 要为True
        :return: output, (hn, cn)
        """
        src_input = self.token_embedding(src_input)  # [batch_size, src_len, embedding_size]
        output, final_state = self.rnn(src_input)
        return output, final_state  # output shape: [batch_size, src_len, hidden_size]


class AttentionDecoder(nn.Module):
    """带有注意力机制解码器的基本接口"""

    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError


class StandardDecoder(AttentionDecoder):
    """
    标准解码器
    """

    def __init__(self, embedding_size, hidden_size, num_layers,
                 rnn_cell=None, batch_first=True, dropout=0.):
        """

        :param embedding_size:
        :param hidden_size:
        :param num_layers: RNN层数
        :param cell_type:
        :param batch_first:
        """
        super(StandardDecoder, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout

        self.rnn = rnn_cell(self.embedding_size, self.hidden_size, num_layers=self.num_layers,
                            batch_first=self.batch_first, dropout=self.dropout)

    def forward(self, tgt_input=None, decoder_state=None, encoder_output=None):
        """

        :param tgt_input: [batch_size, tgt_len, embedding_size] 这种情况 batch_first 要为True
        :param decoder_state: encoder的state, 包含(hn, cn)两部分，这里就决定了encoder和decoder的hidden_size要一致
                              解码第一个时刻的时候，decoder_state为编码器最后一个时刻的encoder_state
        :return: output, (hn, cn)
        """
        output, final_state = self.rnn(tgt_input, decoder_state)
        return output, final_state


class LuongAttentionDecoder(AttentionDecoder):
    """
    Luong's multiplicative attention
    """

    def __init__(self, embedding_size, hidden_size, num_layers,
                 rnn_cell=None, batch_first=True, dropout=0.):
        super(LuongAttentionDecoder, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.rnn = rnn_cell(self.embedding_size + self.hidden_size,
                            self.hidden_size, num_layers=self.num_layers,
                            batch_first=self.batch_first, dropout=self.dropout)

    def forward(self, tgt_input=None, decoder_state=None, encoder_output=None):
        """

        :param tgt_input: [batch_size, tgt_len, embedding_size] 这种情况 batch_first 要为True
        :param decoder_state (hn,cn): 在解码第1个时刻时，decoder_state为encoder最后一个时刻的状态
                              后续则为decoder上一个时刻的状态
                hn: [num_layer, batch_size, hidden_size]
                cn: [num_layer, batch_size, hidden_size]
        :param encoder_output: encoder最后一层所有时刻的输出, [batch_size, time_step, hidden_size]
        :return:
        """
        tgt_input = tgt_input.permute(1, 0, 2)  # [tgt_len, batch_size, embedding_size]
        outputs, self._attention_weights = [], []
        for tgt_in in tgt_input:  # 开始遍历每个时刻
            pass

    def attention(self, query, key, value, padding_idx=0):
        """

        :param query:  hidden_state中的hn的最后一层: [batch_size, hidden_size]
        :param key:    encoder_output [batch_size, time_step, hidden_size]
        :param value:  encoder_output [batch_size, time_step, hidden_size]
        :param padding_idx:  填充值
        :return:
        """
        scores = torch.bmm(self.linear(query).unsqueeze(1), key.transpose(1, 2))  # [batch_size, tgt_len, time_step]

        # [batch_size, hidden_size] @ [hidden_size, hidden_size] = [batch_size, hidden_size]
        # [batch_size, 1, hidden_size] @ [batch_size, hidden_size, time_step]= [batch_size, 1, time_step]
        pass

    @property
    def attention_weights(self):
        return self._attention_weights


class DecoderWrapper(nn.Module):
    """
    解码器
    """

    def __init__(self, embedding_size, hidden_size, num_layers, vocab_size,
                 cell_type='LSTM', decoder_type='standard', batch_first=True, dropout=0.):
        """

        :param embedding_size:
        :param hidden_size:
        :param num_layers: RNN层数
        :param vocab_size:
        :param cell_type:
        :param batch_first:
        """
        super(DecoderWrapper, self).__init__()
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.cell_type = cell_type
        self.decoder_type = decoder_type

        if cell_type == 'LSTM':
            rnn_cell = nn.LSTM
        elif cell_type == 'GRU':
            rnn_cell = nn.GRU
        else:
            raise ValueError("Unrecognized RNN cell type: " + cell_type)

        self.token_embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        if self.decoder_type == 'standard':
            self.decoder_wrapper = StandardDecoder(embedding_size, hidden_size, num_layers,
                                                   rnn_cell, batch_first=batch_first, dropout=dropout)

    def forward(self, tgt_input=None, decoder_state=None, encoder_output=None):
        """

        :param tgt_input: [batch_size, tgt_len] 这种情况 batch_first 要为True
        :param decoder_state: state, 包含(hn, cn)两部分，这里就决定了encoder和decoder的hidden_size要一致
                               解码第一个时刻的时候，decoder_state为编码器最后一个时刻的state
        :return: output, (hn, cn)
        """
        tgt_input = self.token_embedding(tgt_input)  # [batch_size, tgt_input, embedding_size]
        output, final_state = self.decoder_wrapper(tgt_input, decoder_state, encoder_output)
        return output, final_state


class Seq2Seq(nn.Module):
    def __init__(self, config=None):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(config.src_emb_size, config.hidden_size, config.num_layers,
                               config.src_v_size, config.cell_type, config.batch_first,
                               config.dropout)
        self.decoder = DecoderWrapper(config.tgt_emb_size, config.hidden_size, config.num_layers,
                                      config.tgt_v_size, config.cell_type, config.decoder_type,
                                      config.batch_first, config.dropout)

    def forward(self, src_input, tgt_input):
        """

        :param src_input: [batch_size, src_len]
        :param tgt_input: [batch_size, tgt_len]
        :return: [batch_size, tgt_len, hidden_size]
        """
        encoder_output, encoder_state = self.encoder(src_input)
        decoder_output, decoder_state = self.decoder(tgt_input, encoder_state, encoder_output)
        return decoder_output
