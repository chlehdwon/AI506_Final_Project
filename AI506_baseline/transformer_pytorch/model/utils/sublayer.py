import torch.nn as nn
from .layer_norm import LayerNorm


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, kv=None):
        "Apply residual connection to any sublayer with the same size."
        if kv == None:
          return x + self.dropout(sublayer(self.norm(x)))
        else:
          return x + self.dropout(sublayer(self.norm(x), self.norm(kv)))
        
class SublayerConnectionEmbed(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnectionEmbed, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, embed):
        "Apply residual connection to any sublayer with the same size."
        sublayer_input = self.norm(x)
        sublayer_input[:, 1:, :] = sublayer_input[:, 1:, :] + embed
        return x + self.dropout(sublayer(sublayer_input))


