import torch


def greedy_decode(model, src_in, start_symbol=2, end_symbol=3, device=None):
    """
    贪婪搜索
    :param model:
    :param src_input: 原始输入，形状为 [1,src_len]
    :param start_symbol: 开始符 <BOS> 对应的 索引ID
    :param end_symbol:   结束符 <EOS> 对应的 索引ID
    :param device:
    :return:
    """
    _, thought_vec = model.encoder(src_in)
    max_len = src_in.shape[1] * 2  # 最大长度取输入的两倍
    tgt_in = torch.LongTensor([[start_symbol]]).to(device)
    for i in range(max_len):
        output = model.decoder(tgt_in, thought_vec)  # [1,current_tgt_len, hidden_size]
        logits = model.classifier(output)  # [1,current_tgt_len, vocab_size]
        all_pred_logits = logits[0][-1]  # [1,vocab_size] , 只取当前时刻的预测输出
        pred = all_pred_logits.argmax()  # 预测当前时刻的结果
        if pred.item() == end_symbol:
            break
        tgt_in = torch.cat([tgt_in, torch.LongTensor([[pred]])], dim=1)  # 拼接
    return tgt_in[:, 1:].tolist()[0]  # 去掉第1个开始符，形状为 [1,tgt_len]
