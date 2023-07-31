import logging
from collections import Counter
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from tqdm import tqdm


def my_tokenizer():
    """
    分词器
    pip install de_core_news_sm-3.0.0.tar.gz
    pip install en_core_web_sm-3.0.0.tar.gz
    :return:
    e.g.
    tokenizer['src']("Zwei junge weiße Männer sind im, Freien in der Nähe vieler Büsche.)
    ['Zwei', 'junge', 'weiße', 'Männer', 'sind', 'im', ',', 'Freien', 'in', 'der', 'Nähe', 'vieler', 'Büsche', '.']
    """
    tokenizer = {}
    # 源序列分词器
    tokenizer['src'] = get_tokenizer('spacy', language='de_core_news_sm')  # 德语
    # 目标序列分词器
    tokenizer['tgt'] = get_tokenizer('spacy', language='en_core_web_sm')  # 英语
    return tokenizer


class Vocab(object):
    """
    根据给定文本路径和参数构建词表
    vocab = Vocab()
    print(vocab.itos)  # 得到一个列表，返回词表中的每一个词；
    print(vocab.itos[num])  # 通过索引返回得到词表中对应的词；
    print(vocab.stoi)  # 得到一个字典，返回词表中每个词的索引；
    print(vocab.stoi['[UNK]'])  # 通过单词返回得到词表中对应的索引
    print(len(vocab))  # 返回词表长度
    :param top_k:  取出现频率最高的前top_k个token
    :param data: 为一个列表，每个元素为一句文本
    :return:
    """

    def __init__(self, tokenizer, file_path, min_freq=5, top_k=None, specials=None):
        """
        根据给定参数构建词表，其中：
                    当top_k不为None时，则min_freq参数无效，取前top_k个词构建词表
                    当top_k为None时，则以min_freq进行过滤并构建词表

        :param tokenizer: 指定的分词器
        :param file_path: 训练集路径
        :param min_freq: 最小词频
        :param top_k: 取出现频率最高的前top_k个token
        :param specials:
        """
        logging.info(f" ## 正在根据训练集构建词表……")
        if specials is None:
            specials = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        self.specials = specials
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.min_freq = min_freq
        self.top_k = top_k
        self.stoi = {specials[0]: 0, specials[1]: 1, specials[2]: 2, specials[3]: 3}
        self.itos = [specials[0], specials[1], specials[2], specials[3]]
        self._build_vocab()

    def _build_vocab(self):
        counter = Counter()
        with open(self.file_path, encoding='utf8') as f:
            for string_ in f:  # 遍历原始文本中的每一行
                string_ = string_.strip()
                counter.update(self.tokenizer(string_))  # 统计每个token出现的频率
        if self.top_k is not None:
            # 如果指定了top_k则取前top_k个词来构建词表
            top_k_words = counter.most_common(self.top_k - len(self.specials))
            # 取前top_k - len(specials) 个，加上UNK和PAD，一共top_k个
        else:
            top_k_words = counter.most_common()  # 整体排序
        for i, word in enumerate(top_k_words):
            if word[1] < self.min_freq and self.top_k is None:
                # 此时以 min_freq 进行判断
                break
            self.stoi[word[0]] = i + len(self.specials)  # len(self.specials)表示已有的特殊字符
            self.itos.append(word[0])
        logging.info(f" ## 词表构建完毕，前100个词为: {list(self.stoi.items())[:100]}")

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get(self.specials[1]))

    def __len__(self):
        return len(self.itos)


class LoadEnglishGermanDataset():
    def __init__(self, train_file_paths=None, batch_size=2, min_freq=2, top_k=None):
        # 根据训练预料建立英语和德语各自的字典

        self.specials = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        self.tokenizer = my_tokenizer()
        self.de_vocab = Vocab(self.tokenizer['src'], file_path=train_file_paths['src'],
                              min_freq=min_freq, top_k=top_k, specials=self.specials)
        self.en_vocab = Vocab(self.tokenizer['tgt'], file_path=train_file_paths['tgt'],
                              min_freq=min_freq, top_k=top_k, specials=self.specials)
        self.PAD_IDX = self.de_vocab['<PAD>']
        self.BOS_IDX = self.de_vocab['<BOS>']
        self.EOS_IDX = self.de_vocab['<EOS>']
        self.batch_size = batch_size

    def data_process(self, file_paths):
        """
        将每一句话中的每一个词根据字典转换成索引的形式
        :param file_paths:
        :return:
        """
        raw_src_iter = iter(open(file_paths['src'], encoding="utf8"))
        raw_tgt_iter = iter(open(file_paths['tgt'], encoding="utf8"))
        data = []
        logging.info(f"### 正在将数据集 {file_paths} 转换成 Token ID ")
        for (raw_src, raw_tgt) in tqdm(zip(raw_src_iter, raw_tgt_iter), ncols=80):
            src_tensor_ = torch.tensor([self.de_vocab[token] for token in
                                       self.tokenizer['src'](raw_src.rstrip("\n"))], dtype=torch.long)
            tgt_tensor_ = torch.tensor([self.en_vocab[token] for token in
                                       self.tokenizer['tgt'](raw_tgt.rstrip("\n"))], dtype=torch.long)
            data.append((src_tensor_, tgt_tensor_))
        # [ (tensor([ 9, 37, 46,  5, 42, 36, 11, 16,  7, 33, 24, 45, 13,  4]), tensor([ 8, 45, 11, 13, 28,  6, 34, 31, 30, 16,  4])),
        #   (tensor([22,  5, 40, 25, 30,  6, 12,  4]), tensor([12, 10,  9, 22, 23,  6, 33,  5, 20, 37, 41,  4])),
        #   (tensor([ 8, 38, 23, 39,  7,  6, 26, 29, 19,  4]), tensor([ 7, 27, 21, 18, 24,  5, 44, 35,  4])),
        #   (tensor([ 8, 21,  7, 34, 32, 17, 44, 28, 35, 20, 10, 41,  6, 15,  4]), tensor([ 7, 29,  9,  5, 15, 38, 25, 39, 32,  5, 26, 17,  5, 43,  4])),
        #   (tensor([ 9,  5, 43, 27, 18, 10, 31, 14, 47,  4]), tensor([ 8, 10,  6, 14, 42, 40, 36, 19,  4]))  ]

        return data

    def load_train_val_test_data(self, train_file_paths, val_file_paths, test_file_paths):
        train_data = self.data_process(train_file_paths)
        val_data = self.data_process(val_file_paths)
        test_data = self.data_process(test_file_paths)
        logging.info(train_data)
        train_iter = DataLoader(train_data, batch_size=self.batch_size,
                                shuffle=True, collate_fn=self.generate_batch)
        valid_iter = DataLoader(val_data, batch_size=self.batch_size,
                                shuffle=True, collate_fn=self.generate_batch)
        test_iter = DataLoader(test_data, batch_size=self.batch_size,
                               shuffle=True, collate_fn=self.generate_batch)
        return train_iter, valid_iter, test_iter

    def generate_batch(self, data_batch):
        """
        自定义一个函数来对每个batch的样本进行处理，该函数将作为一个参数传入到类DataLoader中。
        由于在DataLoader中是对每一个batch的数据进行处理，所以这就意味着下面的pad_sequence操作，最终表现出来的结果就是
        不同的样本，padding后在同一个batch中长度是一样的，而在不同的batch之间可能是不一样的。因为pad_sequence是以一个batch中最长的
        样本为标准对其它样本进行padding
        :param data_batch:
        :return:
        """
        de_batch, en_batch = [], []
        for (de_item, en_item) in data_batch:  # 开始对一个batch中的每一个样本进行处理。
            de_batch.append(de_item)  # 编码器输入序列不需要加起止符
            # 在每个idx序列的首位加上 起始token 和 结束 token
            en = torch.cat([torch.tensor([self.BOS_IDX]), en_item, torch.tensor([self.EOS_IDX])], dim=0)
            en_batch.append(en)
        # 以最长的序列为标准进行填充
        de_batch = pad_sequence(de_batch, padding_value=self.PAD_IDX)  # [de_len,batch_size]
        en_batch = pad_sequence(en_batch, padding_value=self.PAD_IDX)  # [en_len,batch_size]
        return de_batch, en_batch
