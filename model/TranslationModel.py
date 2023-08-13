from .Seq2Seq import Seq2Seq
import torch.nn as nn


class TranslationModel(nn.Module):
    """
    翻译模型
    """

    def __init__(self, config=None):
        super().__init__()
        self.seq2seq = Seq2Seq(config)
        self.classifier = nn.Linear(config.hidden_size, config.tgt_v_size)

    def forward(self, src_input, tgt_input, src_key_padding_mask=None):
        """

        :param src_input:  [batch_size, src_len]
        :param tgt_input:  [batch_size, tgt_len]
        :param src_key_padding_mask:  [batch_size, src_len] 标识src_input中哪些位置是填充值，True表示填充
        :return:
        """
        output = self.seq2seq(src_input, tgt_input, src_key_padding_mask)  # [batch_size, tgt_len, hidden_size]
        logits = self.classifier(output)  # [batch_size, tgt_len,tgt_v_size]
        return logits

    def encoder(self, src_input):
        output, final_state = self.seq2seq.encoder(src_input)
        return output, final_state

    def decoder(self, tgt_input, decoder_state, encoder_output):
        """

        :param tgt_input:
        :param decoder_state:  解码第一个时刻的时候，decoder_state为编码器最后一个时刻的state
        :param encoder_output: 用于计算注意力权重
        :return:
        """
        output, final_state = self.seq2seq.decoder(tgt_input, decoder_state, encoder_output)
        return output, final_state
