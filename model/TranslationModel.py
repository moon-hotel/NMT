from .Seq2Seq import Seq2Seq
import torch.nn as nn


class TranslationModel(nn.Module):
    """
    翻译模型
    """

    def __init__(self, config = None):
        super().__init__()
        self.seq2seq = Seq2Seq(config)
        self.classifier = nn.Linear(config.hidden_size, config.tgt_v_size)

    def forward(self, src_input, tgt_input):
        """

        :param src_input:  [batch_size, src_len]
        :param tgt_input:  [batch_size, tgt_len]
        :return:
        """
        output = self.seq2seq(src_input, tgt_input)  # [batch_size, tgt_len, hidden_size]
        logits = self.classifier(output)  # [batch_size, tgt_len,tgt_v_size]
        return logits

    def encoder(self, src_input):
        output, final_state = self.seq2seq.encoder(src_input)
        return output, final_state

    def decoder(self, tgt_input, encoder_state):
        output, _ = self.seq2seq.decoder(tgt_input, encoder_state)
        return output
