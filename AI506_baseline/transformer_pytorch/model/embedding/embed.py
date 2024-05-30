import torch.nn as nn
import torch

class EmbeddingSUM(nn.Module):
    def __init__(self, product_vocab_size, customer_vocab_size, color_vocab_size, size_vocab_size, group_vocab_size, embed_size=768, dropout=0.1):
        super().__init__()
        self.product_token = nn.Embedding(product_vocab_size, embed_size, padding_idx=0)
        self.customer_token = nn.Embedding(customer_vocab_size, embed_size, padding_idx=0)
        self.color_token = nn.Embedding(color_vocab_size, embed_size, padding_idx=0)
        self.size_token = nn.Embedding(size_vocab_size, embed_size, padding_idx=0)
        self.group_token = nn.Embedding(group_vocab_size, embed_size, padding_idx=0)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, product, customer, color, size, group):
        x = self.product_token(product) + self.customer_token(customer) + self.color_token(color) + self.size_token(size) + self.group_token(group)
        return self.dropout(x)

class EmbeddingCC(nn.Module):
    def __init__(self, product_vocab_size, customer_vocab_size, color_vocab_size, size_vocab_size, group_vocab_size, embed_size=768, dropout=0.1):
        super().__init__()
        self.product_token = nn.Embedding(product_vocab_size, embed_size, padding_idx=0)
        self.customer_token = nn.Embedding(customer_vocab_size, embed_size, padding_idx=0)
        self.color_token = nn.Embedding(color_vocab_size, embed_size // 3, padding_idx=0)
        self.size_token = nn.Embedding(size_vocab_size, embed_size // 3, padding_idx=0)
        self.group_token = nn.Embedding(group_vocab_size, embed_size // 3, padding_idx=0)

        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, product, customer, color, size, group):
        x = torch.cat([self.product_token(product), self.customer_token(customer), self.color_token(color), self.size_token(size), self.group_token(group)], dim=-1)
        return self.dropout(x)
    
    