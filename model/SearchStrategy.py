import logging

import torch


def greedy_decode(model, src_in, start_symbol=2, end_symbol=3, device=None):
    """
    贪婪搜索
    :param model:
    :param src_input: 原始输入，形状为 [1,src_len], 已经根据字典转换为了token序列
    :param start_symbol: 开始符 <BOS> 对应的 索引ID
    :param end_symbol:   结束符 <EOS> 对应的 索引ID
    :param device:
    :return:
    """
    encoder_out, decoder_state = model.encoder(src_in)
    max_len = src_in.shape[1] * 2  # 最大长度取输入的两倍
    tgt_in = torch.LongTensor([[start_symbol]]).to(device)  # [1,1]
    results = []
    for i in range(max_len):
        decoder_out, decoder_state = model.decoder(tgt_in, decoder_state, encoder_out)
        # decoder_out shaep: [1,1, hidden_size]
        logits = model.classifier(decoder_out)  # [1,1, tgt_vocab_size]
        pred = logits.argmax()  # 预测当前时刻的结果, 0-d
        results.append(pred.detach().cpu().item())
        if pred.item() == end_symbol:
            break
        tgt_in = torch.LongTensor([[pred]]).to(device)
    return results  # [tgt_len]
