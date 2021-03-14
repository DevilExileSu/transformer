import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from model.base_model import BaseModel


class WordEmbeddings(nn.Embedding):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__(vocab_size, embedding_dim)


class PositionEncoder(BaseModel):
    def __init__(self, h_dim, device, max_seq_len=200, dropout=0.1):
        super().__init__()
        self.device = device
        self.h_dim = h_dim
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, h_dim)
        for pos in range(max_seq_len):
            for i in range(0, h_dim, 2):
                pe[pos, i]   = math.sin(pos / (10000 ** ((2 * i) / h_dim)))
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / h_dim)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, inputs):
        # make embeddings relatively larger
        inputs = inputs * math.sqrt(self.h_dim)
        # add constant to embedding
        seq_len = inputs.size(1)
        pe = Variable(self.pe[:, :seq_len], requires_grad=False).to(self.device)
        
        inputs = inputs + pe
        return self.dropout(inputs)


class TokenTypeEmbeddings(nn.Embedding):
    def __init__(self, type_vocab_size, embedding_dim):
        super().__init__(type_vocab_size, embedding_dim)
        


class PositionEmbeddings(nn.Embedding):
    def __init__(self, max_position_embeddings, embedding_dim):
        super().__init__(max_position_embeddings, embedding_dim)


# LayerNorm