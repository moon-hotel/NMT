import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, vocab_size,
                 cell_type='LSTM', bidirectional=False, batch_first=True):
        super(Encoder, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.cell_type = cell_type
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        if cell_type == 'RNN':
            rnn_cell = nn.RNN
        elif cell_type == 'LSTM':
            rnn_cell = nn.LSTM
        elif cell_type == 'GRU':
            rnn_cell = nn.GRU
        else:
            raise ValueError("Unrecognized RNN cell type: " + cell_type)

        self.token_embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.rnn = rnn_cell(self.embedding_size, self.hidden_size, num_layers=self.num_layers,
                            batch_first=self.batch_first, bidirectional=self.bidirectional)

    def forward(self, src_input=None):
        src_input = self.token_embedding(src_input)  # [batch_size, src_len, embedding_size]
        output, final_state = self.rnn(src_input)
        return output, final_state


class Decoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, vocab_size,
                 cell_type='LSTM', bidirectional=False, batch_first=True):
        super(Decoder, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.cell_type = cell_type
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        if cell_type == 'RNN':
            rnn_cell = nn.RNN
        elif cell_type == 'LSTM':
            rnn_cell = nn.LSTM
        elif cell_type == 'GRU':
            rnn_cell = nn.GRU
        else:
            raise ValueError("Unrecognized RNN cell type: " + cell_type)

        self.token_embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.rnn = rnn_cell(self.embedding_size, self.hidden_size, num_layers=self.num_layers,
                            batch_first=self.batch_first, bidirectional=self.bidirectional)

    def forward(self, tgt_input=None, encoder_state=None):
        tgt_input = self.token_embedding(tgt_input)  # [batch_size, tgt_input, embedding_size]
        output, final_state = self.rnn(tgt_input, encoder_state)
        return output, final_state


class Seq2Seq(nn.Module):
    def __init__(self, src_emb_size, tgt_emb_size, hidden_size, num_layers, src_v_size, tgt_v_size,
                 cell_type='LSTM', bidirectional=False, batch_first=True):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(src_emb_size, hidden_size, num_layers, src_v_size,
                               cell_type, bidirectional, batch_first)
        self.decoder = Decoder(tgt_emb_size, hidden_size, num_layers, tgt_v_size,
                               cell_type, bidirectional, batch_first)

    def forward(self, src_input, tgt_input):
        encoder_output, encoder_state = self.encoder(src_input)
        decoder_output, _ = self.decoder(tgt_input, encoder_state)
        return decoder_output
