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


def test_TranslationModelInference():
    src_input = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    tgt_input = torch.LongTensor([[1]])
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
    _, thought_vec = model.encoder(src_input)
    for i in range(10):
        output = model.decoder(tgt_input, thought_vec)  # [1,current_tgt_len, vocab_size]
        logits = model.classifier(output)
        print(logits.shape)
        all_pred_logits = logits[0][-1]  # [1,vocab_size]
        pred = all_pred_logits.argmax()
        tgt_input = torch.cat([tgt_input, torch.LongTensor([[pred]])], dim=1)
        print(tgt_input.shape)


if __name__ == '__main__':
    # test_TranslationModel()
    test_TranslationModelInference()
