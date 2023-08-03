import logging
import torch
import os
from model import TranslationModel
from config import TranslationModelConfig
from utils import LoadEnglishGermanDataset
from torchtext.data.metrics import bleu_score


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
    model = TranslationModel(config)
    if os.path.exists(config.model_save_path):
        loaded_paras = torch.load(config.model_save_path)
        model.load_state_dict(loaded_paras)
        logging.info("#### 成功载入已有模型，追加训练...")
    model = model.to(config.devices[0])
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=data_loader.TGT_PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    logging.info("############开始训练############")
    for epoch in range(config.epochs):
        for i, (src_input, tgt_input) in enumerate(train_iter):
            src_input, tgt_input = src_input.to(config.devices[0]), tgt_input.to(config.devices[0])
            tgt_in = tgt_input[:, :-1]  # 注意，这样的索引方式是batch_first = True时，如果False则为 [:-1,:]
            tgt_out = tgt_input[:, 1:]  # # [batch_size, tgt_out_len]
            logits = model(src_input, tgt_in)  # [batch_size, tgt_out_len, vocab_size]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                logging.info(f"Epochs[{epoch + 1}/{config.epochs}]--batch[{i}/{len(train_iter)}]"
                             f"--ppl: {round(torch.exp(loss).item(), 4)}--loss: {round(loss.item(), 4)}")
                logging.info(f"bleu: {round(compute_bleu(logits, tgt_out), 4)}")
                # logging.info(f"{bleu_score()}")
                # writer.add_scalar('Training/Accuracy', acc, scheduler.last_epoch)
            # writer.add_scalar('Training/Loss', loss.item(), scheduler.last_epoch)
        # test_acc = evaluate(val_iter, model, config.device)
        # test_acc = evaluate(train_iter, model, config.device)
        # logging.info(f"Epochs[{epoch + 1}/{config.epochs}]--Acc on val {test_acc}")
        # writer.add_scalar('Testing/Accuracy', test_acc, scheduler.last_epoch)
        # if test_acc > max_test_acc:  # 因为
        #     max_test_acc = test_acc
        #     state_dict = deepcopy(model.state_dict())
        #     torch.save(state_dict, config.model_save_path)


def compute_bleu(logits, y_true):
    """

    :param logits: [batch_size, tgt_out_len]
    :param y_true: [batch_size, tgt_out_len]
    :return:
    """
    y_pred, y_true = logits.argmax(dim=-1).tolist(), y_true.tolist()
    y_pred = [[str(item) for item in x] for x in y_pred]
    y_true = [[[str(item) for item in x]] for x in y_true]
    return bleu_score(y_pred, y_true, max_n=4)


if __name__ == '__main__':
    config = TranslationModelConfig()
    train(config)
    # y_pred = torch.tensor([[1, 2, 3, 4, 5, 6], [2, 2, 2, 3, 5, 6]])
    # y_true = torch.tensor([[1, 2, 3, 4, 5, 6], [2, 2, 2, 3, 5, 7]])

