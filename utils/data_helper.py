from collections import Counter
import torch
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from tqdm import tqdm
import logging
import os
from .tools import process_cache
from copy import deepcopy

PROJECT_HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_HOME = os.path.join(PROJECT_HOME, 'data')


def my_tokenizer():
    """
    分词器
    本示例中使用到的tokenizer需要安装spacy包和下面这两个包，data目录下有提供
    pip install de_core_news_sm-3.6.0.tar.gz
    pip install en_core_web_sm-3.6.0.tar.gz

    参考下载地址
    https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.6.0/de_core_news_sm-3.6.0-py3-none-any.whl
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.6.0/en_core_web_sm-3.6.0-py3-none-any.whl
    :return:
    e.g.
    tokenizer['src']("Zwei junge weiße Männer sind im, Freien in der Nähe vieler Büsche.)
    ['Zwei', 'junge', 'weiße', 'Männer', 'sind', 'im', ',', 'Freien', 'in', 'der', 'Nähe', 'vieler', 'Büsche', '.']
    """
    tokenizer = {}
    # 源序列分词器

    # 这里如果是换成别的平行语料，则需要在这里实现对应的分词器
    tokenizer['src'] = get_tokenizer('spacy', language='de_core_news_sm')  # 源序列（这里是德语）
    # 目标序列分词器
    tokenizer['tgt'] = get_tokenizer('spacy', language='en_core_web_sm')  # 目标序列（这里是英语）
    return tokenizer


def pad_sequence(sequences, batch_first=False, max_len=None, padding_value=0):
    """
    对一个List中的元素进行padding
    Pad a list of variable length Tensors with ``padding_value``
    a = torch.ones(25)
    b = torch.ones(22)
    c = torch.ones(15)
    pad_sequence([a, b, c],max_len=None).size()
    torch.Size([25, 3])

    sequence: 为list, 每个元素为一个tensor
    batch_first: 是否把batch_size放到第一个维度
    padding_value: padding值
    max_len :
            当max_len = 50时，表示以某个固定长度对样本进行padding，如有多余则截掉；
            当max_len=None是，表示以当前batch中最长样本的长度对其它进行padding。
    """
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    out_tensors = []
    for tensor in sequences:
        if tensor.size(0) < max_len:
            padding_content = [padding_value] * (max_len - tensor.size(0))
            tensor = torch.cat([tensor, torch.tensor(padding_content)], dim=0)
        else:
            tensor = tensor[:max_len]
        out_tensors.append(tensor)
    out_tensors = torch.stack(out_tensors, dim=1)
    if batch_first:
        return out_tensors.transpose(0, 1)
    return out_tensors


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
        self.stoi = {token: idx for idx, token in enumerate(specials)}  # str to index
        self.itos = specials[::]  # 相当于深拷贝， index to str
        self.build_vocab()

    @process_cache(unique_key=['min_freq', 'top_k'])
    def _build_vocab(self, file_path=None):
        """
        根据语料构建词表，并会将结果缓存至file_path所在的目录
        :param file_path: 语料路径
        :return:
        """
        counter = Counter()
        with open(file_path, encoding='utf8') as f:
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
        logging.info(f" ## 词表构建完毕，总长度为{len(self.itos)}, "
                     f"前100个词为: {list(self.stoi.items())[:100]}")
        return self

    def build_vocab(self):
        vocab = self._build_vocab(file_path=self.file_path)
        if self is not vocab:
            # 当前情况为载入本地缓存的词表类对象
            for k, v in self.__dict__.items():
                # 将之前本地缓存的词表类对象的属性复制到当前对象
                self.__dict__[k] = deepcopy(vocab.__dict__[k])
        del vocab

    def __getitem__(self, token):
        """
        通过以vocab['word']的方式来获取词'word'对应的索引，如果不存在于词表中则返回'<UNK>'对应的索引
        其中 vocab = Vocab(),
        :param token:
        :return:
        """
        return self.stoi.get(token, self.stoi.get(self.specials[1]))

    def __len__(self):
        """
        得到词表长度
        :return:
        """
        return len(self.itos)


class LoadEnglishGermanDataset():
    """
    构建德语（src）-英语（tgt）平行语料
    """
    DATA_DIR = os.path.join(DATA_HOME, 'GermanEnglish')
    DATA_FILE_PATH = {'train': {'src': os.path.join(DATA_DIR, 'train.de'),
                                'tgt': os.path.join(DATA_DIR, 'train.en')},
                      'dev': {'src': os.path.join(DATA_DIR, 'val.de'),
                              'tgt': os.path.join(DATA_DIR, 'val.en')},
                      'test': {'src': os.path.join(DATA_DIR, 'test.de'),
                               'tgt': os.path.join(DATA_DIR, 'test.en')}}
    # 指定数据集对应的路径
    CACHE_FILE_PATH = {'train': os.path.join(DATA_DIR, 'train'),
                       'dev': os.path.join(DATA_DIR, 'dev'),
                       'test': os.path.join(DATA_DIR, 'test')}

    # 指定数据集构建完毕后的缓存路径

    def __init__(self, batch_size=2, min_freq=2, src_top_k=None,
                 tgt_top_k=None, src_inverse=True, batch_first=True):
        """

        :param batch_size:
        :param min_freq:
        :param src_top_k: 对于源输入来说，如果src_top_k=None，则以min_freq进行过滤
        :param tgt_top_k: 对于目标输入来说，同上。 但是两者相互独立，例如src_top_k=None,
                          tgt_top_k= 500, min_freq=10, 则表示源输入以min_freq进行过滤
                          目标输入以tgt_top_k进行过滤
        :param src_inverse:  是否将源输入反转
        :param batch_first:
        """
        self.batch_size = batch_size
        self.min_freq = min_freq
        self.tgt_top_k = tgt_top_k
        self.src_top_k = src_top_k
        self.src_inverse = src_inverse
        self.batch_first = batch_first
        self.tokenizer = my_tokenizer()
        logging.info(f" ## 构建源输入词表......")
        self.src_vocab = Vocab(self.tokenizer['src'], file_path=self.DATA_FILE_PATH['train']['src'],
                               min_freq=min_freq, top_k=src_top_k, specials=['<PAD>', '<UNK>'])
        logging.info(f" ## 构建目标输入词表......")
        self.tgt_vocab = Vocab(self.tokenizer['tgt'], file_path=self.DATA_FILE_PATH['train']['tgt'],
                               min_freq=min_freq, top_k=tgt_top_k,
                               specials=['<PAD>', '<UNK>', '<BOS>', '<EOS>'])
        self.TGT_PAD_IDX = self.tgt_vocab['<PAD>']
        self.TGT_BOS_IDX = self.tgt_vocab['<BOS>']
        self.TGT_EOS_IDX = self.tgt_vocab['<EOS>']
        self.SRC_PAD_IDX = self.src_vocab['<PAD>']

    @process_cache(unique_key=['min_freq', 'src_top_k', 'tgt_top_k', 'batch_first'])
    def data_process(self, file_path=None):
        """
        将训练集或验证集或测试集中的平行语料文本转换成token
        此处的file_path实际上指定的是构建完毕后的缓存路径
        :param file_path:
        :return:
        """
        data_name = file_path.split(os.sep)[-1]
        # 指定源输入和目标输入各自的路径
        raw_src_iter = iter(open(self.DATA_FILE_PATH[data_name]['src'], encoding="utf8"))
        raw_tgt_iter = iter(open(self.DATA_FILE_PATH[data_name]['tgt'], encoding="utf8"))
        data = []
        logging.info(f"### 正在将数据集 {file_path} 转换成 Token ID ")
        for (raw_src, raw_tgt) in tqdm(zip(raw_src_iter, raw_tgt_iter), ncols=80):
            src_tokens = self.tokenizer['src'](raw_src.rstrip("\n"))
            logging.debug(f"src_tokens: {src_tokens}")
            src_tensor_ = torch.tensor([self.src_vocab[token] for token in src_tokens], dtype=torch.long)
            logging.debug(f"src_token_ids: {src_tensor_}")

            tgt_tokens = self.tokenizer['tgt'](raw_tgt.rstrip("\n"))
            logging.info(f"tgt_tokens: {tgt_tokens}")
            tgt_tensor_ = torch.tensor([self.tgt_vocab[token] for token in tgt_tokens], dtype=torch.long)
            logging.info(f"tgt_token_ids: {tgt_tensor_}")
            data.append((src_tensor_, tgt_tensor_))
        # [ (tensor([ 9, 37, 46,  5, 42, 36, 11, 16,  7, 33, 24, 45, 13,  4]), tensor([ 8, 45, 11, 13, 28,  6, 34, 31, 30, 16,  4])),
        #   (tensor([22,  5, 40, 25, 30,  6, 12,  4]), tensor([12, 10,  9, 22, 23,  6, 33,  5, 20, 37, 41,  4])),
        #   (tensor([ 8, 38, 23, 39,  7,  6, 26, 29, 19,  4]), tensor([ 7, 27, 21, 18, 24,  5, 44, 35,  4])),
        #   (tensor([ 8, 21,  7, 34, 32, 17, 44, 28, 35, 20, 10, 41,  6, 15,  4]), tensor([ 7, 29,  9,  5, 15, 38, 25, 39, 32,  5, 26, 17,  5, 43,  4])),
        #   (tensor([ 9,  5, 43, 27, 18, 10, 31, 14, 47,  4]), tensor([ 8, 10,  6, 14, 42, 40, 36, 19,  4]))  ]

        return data

    def load_train_val_test_data(self, is_train=False):
        """
        载入对应的迭代器
        :param is_train:
        :return:
        """
        if not is_train:
            test_data = self.data_process(self.CACHE_FILE_PATH['test'])
            test_iter = DataLoader(test_data, batch_size=self.batch_size,
                                   shuffle=False, collate_fn=self.generate_batch)
            return test_iter

        train_data = self.data_process(file_path=self.CACHE_FILE_PATH['train'])
        val_data = self.data_process(file_path=self.CACHE_FILE_PATH['dev'])
        train_iter = DataLoader(train_data, batch_size=self.batch_size,
                                shuffle=True, collate_fn=self.generate_batch)
        valid_iter = DataLoader(val_data, batch_size=self.batch_size,
                                shuffle=False, collate_fn=self.generate_batch)
        return train_iter, valid_iter

    def generate_batch(self, data_batch):
        """
        自定义一个函数来对每个batch的样本进行处理，该函数将作为一个参数传入到类DataLoader中。
        由于在DataLoader中是对每一个batch的数据进行处理，所以这就意味着下面的pad_sequence操作，最终表现出来的结果就是
        不同的样本，padding后在同一个batch中长度是一样的，而在不同的batch之间可能是不一样的。因为pad_sequence是以一个batch中最长的
        样本为标准对其它样本进行padding
        :param data_batch:
        :return:
        """
        src_batch, tgt_batch = [], []
        for (src_item, tgt_item) in data_batch:  # 开始对一个batch中的每一个样本进行处理。
            if self.src_inverse:  # 逆序输出
                src_item = torch.flip(src_item, dims=[0])
            src_batch.append(src_item)  # 编码器输入序列不需要加起止符
            # 在每个idx序列的首位加上 起始token 和 结束 token
            tgt_item = torch.cat([torch.tensor([self.TGT_BOS_IDX]), tgt_item,
                                  torch.tensor([self.TGT_EOS_IDX])], dim=0)
            tgt_batch.append(tgt_item)
        # 以最长的序列为标准进行填充
        src_batch = pad_sequence(src_batch, self.batch_first, padding_value=self.TGT_PAD_IDX)  # [src_len,batch_size]
        tgt_batch = pad_sequence(tgt_batch, self.batch_first, padding_value=self.TGT_PAD_IDX)  # [tgt_len,batch_size]
        return src_batch, tgt_batch
