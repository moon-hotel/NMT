"""
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import logging
import torch
import os
from copy import deepcopy
from model import TranslationModel
from model import greedy_decode
from config import TranslationModelConfig
from utils import LoadEnglishGermanDataset
from torchtext.data.metrics import bleu_score
from transformers import get_polynomial_decay_schedule_with_warmup


def train(config=None):
    logging.info("############载入数据集############")
    data_loader = LoadEnglishGermanDataset(batch_size=config.batch_size,
                                           min_freq=config.min_freq,
                                           src_top_k=config.src_v_size,
                                           tgt_top_k=config.tgt_v_size,
                                           src_inverse=config.src_inverse)
    train_iter, valid_iter = data_loader.load_train_val_test_data(is_train=True)
    logging.info("############初始化模型############")
    config.src_v_size = len(data_loader.src_vocab)
    config.tgt_v_size = len(data_loader.tgt_vocab)
    config.show_paras()
    model = TranslationModel(config)
    if os.path.exists(config.model_save_path):
        loaded_paras = torch.load(config.model_save_path)
        model.load_state_dict(loaded_paras)
        logging.info("#### 成功载入已有模型，追加训练...")
    model = model.to(config.devices[0])  # 放到主GPU上
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=data_loader.TGT_PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    num_training_steps = len(train_iter) * config.epochs
    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, config.num_warmup_steps,
                                                          num_training_steps, lr_end=1e-7, power=3)
    logging.info("############开始训练############")
    max_bleu = -1
    for epoch in range(config.epochs):
        for i, (src_input, tgt_input) in enumerate(train_iter):
            src_input, tgt_input = src_input.to(config.devices[0]), tgt_input.to(config.devices[0])
            src_key_padding_mask = (src_input == data_loader.SRC_PAD_IDX)
            tgt_in = tgt_input[:, :-1]  # 注意，这样的索引方式是batch_first = True时，如果False则为 [:-1,:]
            tgt_out = tgt_input[:, 1:]  # # [batch_size, tgt_out_len]
            logits = model(src_input, tgt_in, src_key_padding_mask)  # [batch_size, tgt_out_len, vocab_size]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1)) / config.batch_size
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
            scheduler.step()
            if i % 50 == 0:
                bleu = compute_bleu(logits.argmax(dim=-1).detach().cpu().tolist(),
                                    tgt_out.detach().cpu().tolist(), False, data_loader.TGT_PAD_IDX)
                logging.info(f"Epochs[{epoch + 1}/{config.epochs}]--batch[{i}/{len(train_iter)}]"
                             f"--loss: {round(loss.item(), 4)}--ppl: {round(torch.exp(loss).item(), 4)}"
                             f"--bleu: {round(bleu, 4)}")
        if epoch % 3 == 0:
            bleu = evaluate(config, valid_iter, model, data_loader)
            logging.info(f"bleu on valid set: {bleu}")
            if bleu >= max_bleu:
                max_bleu = bleu
                state_dict = deepcopy(model.state_dict())
                torch.save(state_dict, config.model_save_path)


def evaluate(config, valid_iter, model, data_loader):
    """
    评估
    :param config:
    :param valid_iter:
    :param model:
    :param data_loader:
    :return:
    """
    model.eval()
    y_preds, y_trues = [], []
    with torch.no_grad():
        for src_input, tgt_input in valid_iter:
            for src_in, tgt_in in zip(src_input, tgt_input):  # 遍历每一个样本
                src_in = src_in.to(config.devices[0])
                tgt_out = tgt_in[1:]  # 去掉第一个时刻，即开始符
                # 此时 src_in和tgt_in的形状为 [src_len]和[tgt_len]
                y_trues.append(tgt_out.tolist())
                y_pred = greedy_decode(model, src_in.reshape(1, -1), data_loader.TGT_BOS_IDX,
                                       data_loader.TGT_EOS_IDX, config.devices[0])
                y_preds.append(y_pred)
    model.train()
    return compute_bleu(y_preds, y_trues, True, data_loader.TGT_PAD_IDX)


def compute_bleu(y_pred, y_true, inference=False, pad_index=0):
    """
    :param y_pred: [batch_size, tgt_out_len]  二维的list
    :param y_true: [batch_size, tgt_out_len]  二维的list
    :return: e.g.
    y_pred = [[1, 2, 3, 4, 5, 6, 6, 7], [2, 2, 2, 3, 5, 6, 3, 4]]
    y_true = [[1, 2, 3, 4, 5, 6, 0, 0], [2, 2, 2, 3, 5, 7, 0, 0]]

    if not inference:
        compute_bleu(y_pred, y_true) 0.88068
        此时:
              y_pred:   [['1', '2', '3', '4', '5', '6'], ['2', '2', '2', '3', '5', '6']]
    y_true_truncated:  [[['1', '2', '3', '4', '5', '6']], [['2', '2', '2', '3', '5', '7']]]

    if inference:
        compute_bleu(y_pred, y_true) 0.59919
        此时:
               y_pred:  [['1', '2', '3', '4', '5', '6', '6', '7'], ['2', '2', '2', '3', '5', '6', '3', '4']]
    y_true_truncated:  [[['1', '2', '3', '4', '5', '6']], [['2', '2', '2', '3', '5', '7']]]

    """
    y_pred = [[str(item) for item in x] for x in y_pred]
    y_true_truncated = []
    for i, x in enumerate(y_true):  # 对正确标签中的填充值进行处理
        tmp = []
        for item in x:
            if item != pad_index:  # 判断当前时刻的真实标签值是否为填充值
                tmp.append(str(item))
            else:
                break
        y_true_truncated.append([tmp])
        if not inference:  # 如果是训练阶段，在计算bleu时则需要去掉解码器正确标签填充部分的值
            y_pred[i] = y_pred[i][:len(tmp)]
            # 注意：如果是在推理阶段，则不能对预测结果做任何操作
    # if inference:
    #     logging.info(f"y_pred: {y_pred[:5]}")
    #     logging.info(f"y_true_truncated: {y_true_truncated[:5]}")
    return bleu_score(y_pred, y_true_truncated, max_n=4)


if __name__ == '__main__':
    config = TranslationModelConfig()
    train(config)