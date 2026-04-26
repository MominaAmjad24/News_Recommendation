import torch
import torch.nn as nn
import torch.nn.functional as F

class AdditiveAttention(nn.Module):
    def __init__(self, dim, hidden_dim=200):
        super().__init__()
        self.proj = nn.Linear(dim, hidden_dim)
        self.query = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        e = torch.tanh(self.proj(x))
        scores = self.query(e).squeeze(-1)
        weights = F.softmax(scores, dim=-1)
        return torch.bmm(weights.unsqueeze(1), x).squeeze(1)

#News Encoder
class NewsEncoder(nn.Module):
    def __init__(self, embedding_matrix, num_heads=8, head_dim=16):
        super().__init__()

        embed_dim = embedding_matrix.shape[1]
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix), freeze=False
        )

        self.proj = nn.Linear(embed_dim, num_heads * head_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=num_heads * head_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.additive = AdditiveAttention(num_heads * head_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.proj(x)
        x, _ = self.attn(x, x, x)
        return self.additive(x)

#User Encoder
class UserEncoder(nn.Module):
    def __init__(self, news_dim, num_heads=8):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=news_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.additive = AdditiveAttention(news_dim)

    def forward(self, x):
        x, _ = self.attn(x, x, x)
        return self.additive(x)

# Full Model
class NRMSModel(nn.Module):
    def __init__(self, embedding_matrix, num_heads=8, head_dim=16):
        super().__init__()

        self.news_encoder = NewsEncoder(embedding_matrix, num_heads, head_dim)

        news_dim = num_heads * head_dim
        self.user_encoder = UserEncoder(news_dim, num_heads)

    def forward(self, history, candidates):
        batch, hist_len, title_len = history.shape
        _, n_cand, _ = candidates.shape

        # encode history
        hist = history.view(-1, title_len)
        hist_vec = self.news_encoder(hist)
        hist_vec = hist_vec.view(batch, hist_len, -1)

        user_vec = self.user_encoder(hist_vec)

        # encode candidates
        cand = candidates.view(-1, title_len)
        cand_vec = self.news_encoder(cand)
        cand_vec = cand_vec.view(batch, n_cand, -1)

        # dot product
        scores = torch.bmm(cand_vec, user_vec.unsqueeze(-1)).squeeze(-1)

        return scores
