"""
Based on Annotated Transformer from Harvard NLP:
https://nlp.seas.harvard.edu/2018/04/03/attention.html#applications-of-attention-in-our-model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from constant import PAD_INDEX
# from config import ARGS
from Model.util_network import get_pad_mask, get_subsequent_mask, clones


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model, bias=False), 4) # Q, K, V, last
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class SAKTLayer(nn.Module):
    """
    Single Encoder block of SAKT
    """
    def __init__(self, hidden_dim, num_head, dropout):
        super().__init__()
        self._self_attn = MultiHeadedAttention(num_head, hidden_dim, dropout)
        self._ffn = PositionwiseFeedForward(hidden_dim, hidden_dim, dropout)
        self._layernorms = clones(nn.LayerNorm(hidden_dim, eps=1e-6), 2)

    def forward(self, query, key, mask=None):
        """
        query: question embeddings
        key: interaction embeddings
        """
        # self-attention block
        output = self._self_attn(query=query, key=key, value=key, mask=mask)
        output = self._layernorms[0](key + output)
        # feed-forward block
        output = self._layernorms[1](output + self._ffn(output))
        return output


class SAKT(nn.Module):
    """
    Transformer-based
    all hidden dimensions (d_k, d_v, ...) are the same as hidden_dim
    """
    def __init__(self, hidden_dim, question_num, num_layers, num_head, dropout, seq_size=100):
        super().__init__()
        self._hidden_dim = hidden_dim
        self._question_num = question_num
        self._seq_size = seq_size

        # Blocks
        self._layers = clones(SAKTLayer(hidden_dim, num_head, dropout), num_layers)

        # prediction layer
        self._prediction = nn.Linear(hidden_dim, 1)

        # Embedding layers
        self._positional_embedding = nn.Embedding(seq_size+1, hidden_dim)
        self._interaction_embedding = nn.Embedding(2*question_num+1, hidden_dim)
        self._question_embedding = nn.Embedding(question_num+1, hidden_dim)
        
        self.classifier = self._prediction

    def _transform_interaction_to_question_id(self, interaction):
        """
        get question_id from interaction index
        if interaction index is a number in [0, question_num], then leave it as-is
        if interaction index is bigger than question_num (in [question_num + 1, 2 * question_num]
        then subtract question_num
        interaction: integer tensor of shape (batch_size, sequence_size)
        """
        return interaction - self._question_num * (interaction > self._question_num).long()

    def _get_position_index(self, question_id):
        """
        [0, 0, 0, 4, 12] -> [0, 0, 0, 1, 2]
        """
        batch_size = question_id.shape[0]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        position_indices = []
        for i in range(batch_size):
            non_padding_num = (question_id[i] != 0).sum(-1).item()
            position_index = [0] * (question_id.shape[1] - non_padding_num) + list(range(1, non_padding_num+1))
            position_indices.append(position_index)
        return torch.tensor(position_indices, dtype=int).to(device)

    def forward(self, interaction_id, target_id):
        """
        Query: Question (skill, exercise, ...) embedding
        Key, Value: Interaction embedding + positional embedding
        """
        question_id = self._transform_interaction_to_question_id(interaction_id)
        question_id = torch.cat([question_id[:, 1:], target_id], dim=-1)

        interaction_vector = self._interaction_embedding(interaction_id)
        question_vector = self._question_embedding(question_id)
        position_index = self._get_position_index(question_id)
        position_vector = self._positional_embedding(position_index)

        mask = get_pad_mask(question_id, 0) & get_subsequent_mask(question_id)
        x = interaction_vector + position_vector

        for layer in self._layers:
            x = layer(query=question_vector, key=x, mask=mask)

        output = self._prediction(x)
        output = output[:, -1, :]
        return output,x



# import numpy as np

# import torch
# import torch.nn as nn


# def future_mask(seq_length):
#     future_mask = np.triu(np.ones((1, seq_length)), k=1).astype('bool')
#     return torch.from_numpy(future_mask)


# class FFN(nn.Module):
#     def __init__(self, state_size=200):
#         super(FFN, self).__init__()
#         self.state_size = state_size

#         self.lr1 = nn.Linear(state_size, state_size)
#         self.relu = nn.ReLU()
#         self.lr2 = nn.Linear(state_size, state_size)
#         self.dropout = nn.Dropout(0.2)
    
#     def forward(self, x):
#         x = self.lr1(x)
#         x = self.relu(x)
#         x = self.lr2(x)
#         return self.dropout(x)


# class SAKT(nn.Module):
#     # def __init__(self, n_skill, max_seq=100, embed_dim=200):
#     def __init__(self, hidden_dim, question_num, num_layers, num_head, dropout, seq_size=100):
#         super(SAKT, self).__init__()
#         self.question_num = question_num
#         self.hidden_dim = hidden_dim

#         self.embedding = nn.Embedding(2*question_num+1, self.hidden_dim)
#         self.pos_embedding = nn.Embedding(seq_size+1, self.hidden_dim)
#         self.e_embedding = nn.Embedding(question_num+1, self.hidden_dim)

#         self.multi_att = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_head, dropout=dropout)

#         self.dropout = dropout
#         self.layer_normal = nn.LayerNorm(hidden_dim) 

#         self.ffn = FFN(hidden_dim)
#         self.pred = nn.Linear(hidden_dim, 1)
#         self.sigmoid = nn.Sigmoid()

#         self._reset_parameters()
    
#     def _reset_parameters(self):
#         r"""Initiate parameters in the model."""
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)

#     def forward(self, x, question_ids):
#         device = x.device        
#         # src_pad_mask = (x == 0)
#         # tgt_pad_mask = (question_ids == 0)
#         # mask = src_pad_mask & tgt_pad_mask

#         x = self.embedding(x)
#         pos_id = torch.arange(x.size(1)).unsqueeze(0).to(device)

#         pos_x = self.pos_embedding(pos_id)
#         x = x + pos_x

#         e = self.e_embedding(question_ids)

#         x = x.permute(1, 0, 2) # x: [bs, s_len, embed] => [s_len, bs, embed]
#         e = e.permute(1, 0, 2)
#         att_mask = future_mask(x.size(0)).to(device)
#         att_output, att_weight = self.multi_att(e, x, x, attn_mask=att_mask)
#         att_output = self.layer_normal(att_output + e)
#         att_output = att_output.permute(1, 0, 2) # att_output: [s_len, bs, embed] => [bs, s_len, embed]
#         # print(att_output.shape, att_weight.shape)
#         x = self.ffn(att_output)
#         # x = self.dropout(x)
#         x = self.layer_normal(x + att_output)
#         x = self.pred(x)
#         x = x[:, -1, :]
#         return x