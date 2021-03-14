import torch
import torch.nn as nn
from model.base_model import BaseModel
from utils.util import attention

class MultiHeadAttentionLayer(BaseModel):
    def __init__(self, h_dim, n_heads, dropout, device):
        super().__init__()
        assert h_dim % n_heads == 0

        self.h_dim = h_dim
        self.n_heads = n_heads
        self.d_k = h_dim // n_heads

        self.fc_q = nn.Linear(h_dim, h_dim)
        self.fc_k = nn.Linear(h_dim, h_dim)
        self.fc_v = nn.Linear(h_dim, h_dim)
        
        self.fc_o = nn.Linear(h_dim, h_dim)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        Q = self.fc_q(q).view(batch_size, -1, self.n_heads, self.d_k)    
        K = self.fc_k(k).view(batch_size, -1, self.n_heads, self.d_k)     
        V = self.fc_v(v).view(batch_size, -1, self.n_heads, self.d_k)  

        Q = Q.transpose(1,2)             # [batch_size, n_heads, q_len, d_k]
        K = K.transpose(1,2)             # [batch_size, n_heads, k_len, d_k]
        V = V.transpose(1,2)             # [batch_size, n_heads, v_len, d_k]

        output = attention(Q, K, V, self.d_k, mask, self.dropout)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.h_dim)
        # output = [batch_size, q_len, h_dim]
        output = self.fc_o(output)

        return output