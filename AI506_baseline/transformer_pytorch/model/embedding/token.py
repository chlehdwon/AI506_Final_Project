import torch.nn as nn


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size=130, embed_size=768):
        super().__init__(vocab_size, embed_size)
