import torch.nn as nn


class Encoder(nn.Module):
    """
    解码器
    """

    def __init__(self, embedding_size, hidden_size, num_layers, vocab_size,
                 cell_type='LSTM', bidirectional=False, batch_first=True):
        """
        :param embedding_size:
        :param hidden_size:
        :param num_layers:  RNN的层数
        :param vocab_size:
        :param cell_type:
        :param bidirectional:
        :param batch_first:
        """
        super(Encoder, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.cell_type = cell_type
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        if cell_type == 'LSTM':
            rnn_cell = nn.LSTM
        elif cell_type == 'GRU':
            rnn_cell = nn.GRU
        else:
            raise ValueError("Unrecognized RNN cell type: " + cell_type)

        self.token_embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.rnn = rnn_cell(self.embedding_size, self.hidden_size, num_layers=self.num_layers,
                            batch_first=self.batch_first, bidirectional=self.bidirectional)

    def forward(self, src_input=None):
        """

        :param src_input: [batch_size, src_len] 这种情况 batch_first 要为True
        :return: output, (hn, cn)
        """
        src_input = self.token_embedding(src_input)  # [batch_size, src_len, embedding_size]
        output, final_state = self.rnn(src_input)
        return output, final_state  # output shape: [batch_size, src_len, hidden_size]


class Decoder(nn.Module):
    """
    解码器
    """

    def __init__(self, embedding_size, hidden_size, num_layers, vocab_size,
                 cell_type='LSTM', bidirectional=False, batch_first=True):
        """

        :param embedding_size:
        :param hidden_size:
        :param num_layers: RNN层数
        :param vocab_size:
        :param cell_type:
        :param bidirectional:
        :param batch_first:
        """
        super(Decoder, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.cell_type = cell_type
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        if cell_type == 'LSTM':
            rnn_cell = nn.LSTM
        elif cell_type == 'GRU':
            rnn_cell = nn.GRU
        else:
            raise ValueError("Unrecognized RNN cell type: " + cell_type)

        self.token_embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.rnn = rnn_cell(self.embedding_size, self.hidden_size, num_layers=self.num_layers,
                            batch_first=self.batch_first, bidirectional=self.bidirectional)

    def forward(self, tgt_input=None, encoder_state=None):
        """

        :param tgt_input: [batch_size, tgt_len] 这种情况 batch_first 要为True
        :param encoder_state: encoder的state, 包含(hn, cn)两部分，这里就决定了encoder和decoder的hidden_size要一致
        :return: output, (hn, cn)
        """
        tgt_input = self.token_embedding(tgt_input)  # [batch_size, tgt_input, embedding_size]
        output, final_state = self.rnn(tgt_input, encoder_state)
        return output, final_state


class Seq2Seq(nn.Module):
    def __init__(self, config=None):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(config.src_emb_size, config.hidden_size,
                               config.num_layers, config.src_v_size, config.cell_type,
                               config.bidirectional, config.batch_first)
        self.decoder = Decoder(config.tgt_emb_size, config.hidden_size,
                               config.num_layers, config.tgt_v_size,
                               config.cell_type, config.bidirectional, config.batch_first)

    def forward(self, src_input, tgt_input):
        """

        :param src_input: [batch_size, src_len]
        :param tgt_input: [batch_size, tgt_len]
        :return: [batch_size, tgt_len, hidden_size]
        """
        encoder_output, encoder_state = self.encoder(src_input)
        decoder_output, _ = self.decoder(tgt_input, encoder_state)
        return decoder_output
