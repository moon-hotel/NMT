import sys
import torch

sys.path.append('../')

from model import TranslationModel


class ModelConfig():
    def __init__(self):
        self.src_emb_size = 32
        self.tgt_emb_size = 64
        self.hidden_size = 128
        self.num_layers = 2
        self.src_v_size = 50
        self.tgt_v_size = 60
        self.cell_type = 'LSTM'
        self.bidirectional = False
        self.batch_first = True


def test_TranslationModel():
    src_input = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9],
                                  [1, 2, 3, 3, 3, 4, 2, 1, 1]])
    tgt_input = torch.LongTensor([[1, 2, 6, 7, 8, 9],
                                  [1, 2, 4, 2, 1, 1]])
    config = ModelConfig()
    model = TranslationModel(config)
    logits = model(src_input, tgt_input)
    print(logits.shape)


def test_TranslationModelInference():
    src_input = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    tgt_input = torch.LongTensor([[1]])
    config = ModelConfig()
    model = TranslationModel(config)
    _, thought_vec = model.encoder(src_input)
    for i in range(10):
        output = model.decoder(tgt_input, thought_vec)  # [1,current_tgt_len, hidden_size]
        logits = model.classifier(output)  # [1,current_tgt_len, vocab_size]
        print(f"第{i + 1}个时刻的预测logits: {logits.shape}")
        all_pred_logits = logits[0][-1]  # [1,vocab_size] , 只取当前时刻的预测输出
        pred = all_pred_logits.argmax() # 预测结果
        tgt_input = torch.cat([tgt_input, torch.LongTensor([[pred]])], dim=1) # 拼接
        print(f"第{i + 1}个时刻预测结束后的结果: {tgt_input.shape}")
        # 第10个时刻预测结束后的结果: torch.Size([1, 11])


if __name__ == '__main__':
    test_TranslationModel()
    test_TranslationModelInference()
