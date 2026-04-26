import pandas as pd
import numpy as np
import re
from collections import Counter

def tokenize(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9 ]', '', text)
    return text.split()


def build_vocab(news_df, max_words=50000):
    counter = Counter()

    for title in news_df['title']:
        tokens = tokenize(title)
        counter.update(tokens)

    most_common = counter.most_common(max_words)

    word2idx = {word: i+1 for i, (word, _) in enumerate(most_common)}
    word2idx["<PAD>"] = 0

    return word2idx


def text_to_sequence(text, word2idx, max_len=20):
    tokens = tokenize(text)
    seq = [word2idx.get(w, 0) for w in tokens]

    # pad or truncate
    if len(seq) < max_len:
        seq += [0] * (max_len - len(seq))
    else:
        seq = seq[:max_len]

    return seq


def build_news_dict(news_df, word2idx, max_len=20):
    news_dict = {}

    for _, row in news_df.iterrows():
        news_id = row['news_id']
        title = row['title']

        news_dict[news_id] = text_to_sequence(title, word2idx, max_len)

    return news_dict
