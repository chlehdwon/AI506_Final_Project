import torch.nn as nn

from .transformer import TransformerBlock, TransformerBlockCustomer
from transformer_pytorch.model.embedding import EmbeddingSUM, EmbeddingCC
import torch
from .utils import SublayerConnection, PositionwiseFeedForward

class Task1transformer(nn.Module):
    def __init__(self, 
                     product_vocab_size=0,
                     customer_vocab_size=0,
                     color_vocab_size=0,
                     size_vocab_size=0,
                     group_vocab_size=0,
                     num_label=3,
                     hidden=768, 
                     n_layers=12, 
                     attn_heads=12,
                     embedding_type="cc",
                     dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        if embedding_type == "sum": 
          self.feed_forward_hidden = hidden * 4
        else:
          self.feed_forward_hidden = hidden * 12

        # embedding for BERT, sum of positional, segment, token embeddings
        if embedding_type == "sum":
          self.embedding = EmbeddingSUM(product_vocab_size=product_vocab_size,
                                          customer_vocab_size=customer_vocab_size,
                                          color_vocab_size=color_vocab_size,
                                          size_vocab_size=size_vocab_size,
                                          group_vocab_size=group_vocab_size, 
                                          embed_size=hidden)
        elif embedding_type == "cc":
          self.embedding = EmbeddingCC(product_vocab_size=product_vocab_size,
                                          customer_vocab_size=customer_vocab_size,
                                          color_vocab_size=color_vocab_size,
                                          size_vocab_size=size_vocab_size,
                                          group_vocab_size=group_vocab_size, 
                                          embed_size=hidden)

        # multi-layers transformer blocks, deep network
        if embedding_type == "sum":
          self.transformer_blocks = nn.ModuleList(
              [TransformerBlock(hidden, attn_heads, self.feed_forward_hidden, dropout) for _ in range(n_layers)])
        else:
          self.transformer_blocks = nn.ModuleList(
              [TransformerBlock(3 * hidden, attn_heads, self.feed_forward_hidden, dropout) for _ in range(n_layers)])
           
        self.classifier = nn.Linear(3 * hidden, num_label)

    def forward(self, product, customer, color, size, group, inference=False):
        # attention masking for padded token
        mask = (product > 0).unsqueeze(1).repeat(1, product.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(product, customer, color, size, group)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
      

        x = self.classifier(x[:, 0:1, :])
        
        if inference:
            x = torch.argmax(x, dim=-1)
            return x

        return x.permute(0, 2, 1)
    
class Task1transformer_CP(nn.Module):
    def __init__(self, 
                     product_vocab_size=0,
                     customer_vocab_size=0,
                     color_vocab_size=0,
                     size_vocab_size=0,
                     group_vocab_size=0,
                     num_label=3,
                     hidden=768, 
                     n_layers=12, 
                     attn_heads=12,
                     embedding_type="cc",
                     dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.feed_forward_hidden = hidden * 4

        self.pd_embed = nn.Embedding(product_vocab_size, hidden, padding_idx=0)
        self.cs_embed = nn.Embedding(customer_vocab_size, hidden, padding_idx=0)
        self.color_embed = nn.Embedding(color_vocab_size, hidden, padding_idx=0)
        self.size_embed = nn.Embedding(size_vocab_size, hidden, padding_idx=0)
        self.group_embed = nn.Embedding(group_vocab_size, hidden, padding_idx=0)

        self.embed_mlp = nn.Sequential(
            nn.Linear(hidden, self.feed_forward_hidden),
            nn.SiLU(),
            nn.Linear(self.feed_forward_hidden, self.feed_forward_hidden),
        )

        # multi-layers transformer blocks, deep network

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlockCustomer(hidden, attn_heads, self.feed_forward_hidden, dropout) for _ in range(n_layers)])

           
        self.classifier = nn.Linear(hidden, num_label)

    def forward(self, product, customer, color, size, group, inference=False):
        # attention masking for padded token
        mask = (product > 0).unsqueeze(1).repeat(1, product.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.pd_embed(product)
        x[:, 1:, :] = x[:, 1:, :] + self.color_embed(color) + self.size_embed(size) + self.group_embed(group)

        inter_embed = self.embed_mlp(self.cs_embed(customer))

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, inter_embed, mask)
      

        x = self.classifier(x[:, 0:1, :])
        
        if inference:
            x = torch.argmax(x, dim=-1)
            return x

        return x.permute(0, 2, 1)
    
class Task2transformer(nn.Module):
    def __init__(self, 
                     product_vocab_size=0,
                     customer_vocab_size=0,
                     color_vocab_size=0,
                     size_vocab_size=0,
                     group_vocab_size=0,
                     num_label=2,
                     hidden=768, 
                     n_layers=12, 
                     attn_heads=12,
                     embedding_type="cc",
                     dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        if embedding_type == "sum":
          self.embedding = EmbeddingSUM(product_vocab_size=product_vocab_size,
                                          customer_vocab_size=customer_vocab_size,
                                          color_vocab_size=color_vocab_size,
                                          size_vocab_size=size_vocab_size,
                                          group_vocab_size=group_vocab_size, 
                                          embed_size=hidden)
        elif embedding_type == "cc":
          self.embedding = EmbeddingCC(product_vocab_size=product_vocab_size,
                                          customer_vocab_size=customer_vocab_size,
                                          color_vocab_size=color_vocab_size,
                                          size_vocab_size=size_vocab_size,
                                          group_vocab_size=group_vocab_size, 
                                          embed_size=hidden)
        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])
        
        self.classifier = nn.Linear(hidden, num_label)

    def forward(self, product, customer, color, size, group, inference=False):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        mask = (product > 0).unsqueeze(1).repeat(1, product.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(product, customer, color, size, group)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        
        x = self.classifier(x)
        
        if inference:
            x = torch.argmax(x, dim=-1)
            return x

        return x.permute(0, 2, 1)
    