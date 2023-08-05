import torch
from model import TranslationModel
from model import greedy_decode
from config import TranslationModelConfig
from utils import LoadEnglishGermanDataset
import logging
import os


def translation(model=None, text=None, config=None, data_loader=None):
    src_tokens = data_loader.tokenizer['src'](text)
    src_tokens = [[data_loader.src_vocab[token] for token in src_tokens]]
    src_tokens = torch.LongTensor(src_tokens).to(config.devices[0])  # [1,src_len]
    trans_tokens = greedy_decode(model, src_tokens, data_loader.TGT_BOS_IDX,
                                 data_loader.TGT_EOS_IDX, config.devices[0])
    trans = [data_loader.tgt_vocab.itos[idx] for idx in trans_tokens]
    return " ".join(trans)


def inference(texts=None):
    config = TranslationModelConfig()
    logging.info("############载入数据集############")
    data_loader = LoadEnglishGermanDataset(batch_size=config.batch_size,
                                           min_freq=config.min_freq,
                                           src_top_k=config.src_v_size,
                                           tgt_top_k=config.tgt_v_size,
                                           src_inverse=config.src_inverse)
    config.src_v_size = len(data_loader.src_vocab)
    config.tgt_v_size = len(data_loader.tgt_vocab)
    config.show_paras()
    logging.info("############初始化模型############")
    model = TranslationModel(config)
    if not os.path.exists(config.model_save_path):
        raise ValueError(f"模型不存在：{config.model_save_path}")
    loaded_paras = torch.load(config.model_save_path)
    model.load_state_dict(loaded_paras)
    logging.info("#### 成功载入已有模型，追加训练...")
    model = model.to(config.devices[0])
    for text in texts:
        logging.info(f"原文: {text}")
        logging.info(f"翻译: {translation(model, text, config, data_loader)}\n")


if __name__ == '__main__':
    texts = ["Ein Mann in einem blauen Hemd steht auf einer Leiter und putzt ein Fenster.",
             "Zwei Männer stehen am Herd und bereiten Essen zu."]
    inference(texts)
