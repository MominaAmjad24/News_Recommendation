# data_loader.py

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


'''
def create_training_samples(behaviors_df, news_dict):
    X = []
    y = []

    for _, row in behaviors_df.iterrows():
        if pd.isna(row['impressions']):
            continue

        impressions = row['impressions'].split()

        pos, neg = [], []

        for item in impressions:
            news_id, label = item.split('-')

            if news_id not in news_dict:
                continue

            if label == '1':
                pos.append(news_dict[news_id])
            else:
                neg.append(news_dict[news_id])

        for p in pos:
            for n in neg:
                X.append([p, n])
                y.append(1)

    return np.array(X), np.array(y)
'''

def create_nrms_samples(behaviors_df, news_dict, max_history=50, neg_k=4):
    samples = []

    title_len = len(next(iter(news_dict.values())))


    for _, row in behaviors_df.iterrows():
        if pd.isna(row['impressions']):
            continue

        # history
        history = row['history'].split() if pd.notna(row['history']) else []
        history_vecs = [news_dict[h] for h in history if h in news_dict]
        history_vecs = history_vecs[-max_history:]

        # pad history
        if len(history_vecs) < max_history:
            history_vecs += [[0]*len(next(iter(news_dict.values())))] * (max_history - len(history_vecs))

        impressions = row['impressions'].split()

        pos = [i.split('-')[0] for i in impressions if i.endswith('-1') and i.split('-')[0] in news_dict]
        neg = [i.split('-')[0] for i in impressions if i.endswith('-0') and i.split('-')[0] in news_dict]
        
        #Need exactly neg_k negatives so every sample has same shape
        if len(neg) < neg_k:
            continue

        for p in pos:
            sampled_neg = np.random.choice(neg, size=neg_k, replace=False)

            candidates = [p] + list(sampled_neg)
            candidate_vecs = [news_dict[c] for c in candidates]

            samples.append({
                'history': np.array(history_vecs),
                'candidates': np.array(candidate_vecs),
                'label': 0  # positive is always index 0
            })

    return samples


class NRMSDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        return {
            "history": torch.tensor(sample["history"], dtype=torch.long),
            "candidates": torch.tensor(sample["candidates"], dtype=torch.long),
            "label": torch.tensor(sample["label"], dtype=torch.long)
        }
