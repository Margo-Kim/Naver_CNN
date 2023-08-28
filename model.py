# model.py
import torch.nn as nn
import torch

class SenCNN(nn.Module):
    def __init__(self,vocab_len,dim):
        super().__init__()
        n_filter = 100
        self.static = nn.Embedding(num_embeddings=vocab_len, embedding_dim=dim, _freeze=True)
        self.static.eval()
        self.nonstatic = nn.Embedding(num_embeddings=vocab_len, embedding_dim=dim)
        self.conv = nn.Conv1d(in_channels=dim, out_channels=n_filter, kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=dim, out_channels=n_filter, kernel_size=2)
        self.linear = nn.Linear(n_filter*2, 1)
        self.dropout = nn.Dropout(p=0.5)



    def forward(self, x):
        # x : [bsz, seq_len]
        emb = self.static(x)  # [bsz, seq_len, 300]
        emb = emb.permute(0, 2, 1)  # [bsz, 300, seq_len]
        output = self.conv(emb)   # [bsz, n_filter(100), num_sliding]
        pooled1 = torch.max(output, 2)[0]  # [bsz, n_filter]

        emb = self.nonstatic(x)  # [bsz, seq_len, 300]
        emb = emb.permute(0, 2, 1)  # [bsz, 300, seq_len]
        output = self.conv2(emb)  # [bsz, n_filter(100), num_sliding]
        pooled2 = torch.max(output, 2)[0]  # [bsz, n_filter]

        pooled = torch.cat([pooled1, pooled2], dim=1)  # [bsz, 2*n_filter]
        pooled = self.dropout(pooled)


        y_hat = self.linear(pooled)  # [bsz, n_cls]
        

        return y_hat