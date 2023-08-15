"""
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

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


def inference(texts=None, labels=None):
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
    for i, text in enumerate(texts):
        logging.info(f"原文: {text}")
        logging.info(f"翻译: {translation(model, text, config, data_loader)}")
        if labels is not None and len(texts) == len(labels):
            logging.info(f"答案: {labels[i]}\n")
        else:
            logging.info("\n")


if __name__ == '__main__':
    texts = [
        "Zwei junge weiße Männer sind im, Freien in der Nähe vieler Büsche.",
        "Mehrere Männer mit Schutzhelmen bedienen ein Antriebsradsystem.",
        "Ein kleines Mädchen klettert in ein Spielhaus aus Holz.",
        "Ein Mann in einem blauen Hemd steht auf einer Leiter und putzt ein Fenster.",
        "Zwei Männer stehen am Herd und bereiten Essen zu.",
        "Eine Gruppe von Männern lädt Baumwolle auf einen Lastwagen",
        "Ein Mann schläft in einem grünen Raum auf einem Sofa.",
        "Ein Junge mit Kopfhörern sitzt auf den Schultern einer Frau.",
        "Zwei Männer bauen eine blaue Eisfischerhütte auf einem zugefrorenen See auf",
        "Ein Mann mit beginnender Glatze, der eine rote Rettungsweste trägt, sitzt in einem kleinen Boot."
    ]

    labels = [
        "Two young, White males are outside near many bushes.",
        "Several men in hard hats are operating a giant pulley system.",
        "A little girl climbing into a wooden playhouse.",
        "A man in a blue shirt is standing on a ladder cleaning a window.",
        "Two men are at the stove preparing food.",
        "A group of men are loading cotton onto a truck",
        "A man sleeping in a green room on a couch.",
        "A boy wearing headphones sits on a woman's shoulders.",
        "Two men setting up a blue ice fishing hut on an iced over lake",
        "A balding man wearing a red life jacket is sitting in a small boat."
    ]
    inference(texts, labels)
