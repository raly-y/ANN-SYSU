import os
import re
import jieba
import nltk
from nltk.tokenize import word_tokenize
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
from src.utils import load_jsonl

# 获取项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 下载 NLTK 数据（仅需运行一次）
nltk.download('punkt', quiet=True)


class TranslationDataset(Dataset):
    def __init__(self, data, zh_vocab, en_vocab, max_len):
        self.data = data
        self.zh_vocab = zh_vocab
        self.en_vocab = en_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        zh_text = sample['zh']
        en_text = sample['en']

        # 分词
        zh_tokens = jieba.lcut(zh_text)[:self.max_len]
        en_tokens = word_tokenize(en_text.lower())[:self.max_len]

        # 转换为索引
        zh_indices = [self.zh_vocab.get(token, self.zh_vocab['<unk>']) for token in ['<sos>'] + zh_tokens + ['<eos>']]
        en_indices = [self.en_vocab.get(token, self.en_vocab['<unk>']) for token in ['<sos>'] + en_tokens + ['<eos>']]

        # 填充到最大长度
        zh_indices = zh_indices + [self.zh_vocab['<pad>']] * (self.max_len + 2 - len(zh_indices))
        en_indices = en_indices + [self.en_vocab['<pad>']] * (self.max_len + 2 - len(en_indices))

        return {
            'zh_indices': torch.tensor(zh_indices, dtype=torch.long),
            'en_indices': torch.tensor(en_indices, dtype=torch.long),
            'zh_text': zh_text,
            'en_text': en_text
        }


def clean_text(text):
    """Clean text by removing illegal characters and normalizing."""
    # 移除非法字符（保留中文、英文、标点）
    text = re.sub(r'[^\u4e00-\u9fff\w\s.,!?]', '', text)
    return text.strip()


def build_vocab(texts, min_freq=2, special_tokens=['<pad>', '<sos>', '<eos>', '<unk>']):
    """Build vocabulary from a list of tokenized texts."""
    counter = Counter()
    for text in texts:
        counter.update(text)

    # 过滤低频词
    tokens = [token for token, freq in counter.items() if freq >= min_freq]
    # 创建词典，确保索引连续
    vocab = {token: idx for idx, token in enumerate(special_tokens + tokens)}
    print(f"Vocabulary size: {len(vocab)}, after filtering tokens with freq < {min_freq}")
    return vocab


def load_pretrained_embeddings(vocab, embedding_file, embedding_dim=300):
    """Load pretrained word embeddings."""
    # embeddings = np.random.normal(0, 0.1, (len(vocab), embedding_dim))
    embeddings = np.zeros((len(vocab), embedding_dim), dtype=np.float32)  # 初始化为零
    found_words = 0
    try:
        with open(embedding_file, 'r', encoding='utf-8') as f:
            # 默认跳过 FastText 第一行（词汇量和维度）
            f.readline()  # 跳过第一行，无日志输出
            for line in f:
                tokens = line.strip().split()
                if len(tokens) != embedding_dim + 1:
                    continue  # 跳过格式错误的行
                word = tokens[0]
                if word in vocab:
                    embeddings[vocab[word]] = np.array(tokens[1:], dtype=np.float32)
                    found_words += 1
    except FileNotFoundError:
        print(f"Error: Embedding file {embedding_file} not found")
        raise
    if found_words > 0:
        mean_embedding = np.mean(embeddings[4:], axis=0)  # 跳过 <pad>, <sos>, <eos>, <unk>
        for idx in range(4):  # 为特殊标记赋值
            embeddings[idx] = mean_embedding
    print(f"Found {found_words}/{len(vocab)} words in pretrained embeddings")
    return torch.tensor(embeddings, dtype=torch.float32)


def preprocess_data(config):
    """Preprocess parallel corpus and create DataLoaders."""
    # 加载配置
    train_path = os.path.join(PROJECT_ROOT, config['data']['train_path'])
    valid_path = os.path.join(PROJECT_ROOT, config['data']['valid_path'])
    test_path = os.path.join(PROJECT_ROOT, config['data']['test_path'])
    embedding_path_zh = os.path.join(PROJECT_ROOT, config['data']['embedding_path_zh'])
    embedding_path_en = os.path.join(PROJECT_ROOT, config['data']['embedding_path_en'])
    min_freq = config['data']['min_freq']
    max_len = config['data']['max_len']
    batch_size = config['training']['batch_size']

    # 加载数据集
    train_data = load_jsonl(train_path)
    valid_data = load_jsonl(valid_path)
    test_data = load_jsonl(test_path)

    # 数据清洗
    for data in [train_data, valid_data, test_data]:
        for sample in data:
            sample['zh'] = clean_text(sample['zh'])
            sample['en'] = clean_text(sample['en'])

    # 分词并收集 token
    zh_texts = [jieba.lcut(sample['zh']) for sample in train_data]
    en_texts = [word_tokenize(sample['en'].lower()) for sample in train_data]

    # 构建词典
    zh_vocab = build_vocab(zh_texts, min_freq)
    en_vocab = build_vocab(en_texts, min_freq)

    # 加载预训练词向量
    zh_embeddings = load_pretrained_embeddings(zh_vocab, embedding_path_zh)
    en_embeddings = load_pretrained_embeddings(en_vocab, embedding_path_en)

    # 创建 Dataset 和 DataLoader
    train_dataset = TranslationDataset(train_data, zh_vocab, en_vocab, max_len)
    valid_dataset = TranslationDataset(valid_data, zh_vocab, en_vocab, max_len)
    test_dataset = TranslationDataset(test_data, zh_vocab, en_vocab, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return {
        'train_loader': train_loader,
        'valid_loader': valid_loader,
        'test_loader': test_loader,
        'zh_vocab': zh_vocab,
        'en_vocab': en_vocab,
        'zh_embeddings': zh_embeddings,
        'en_embeddings': en_embeddings
    }
