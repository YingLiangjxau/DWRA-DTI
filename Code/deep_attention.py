import torch
import torch.nn as nn

import torch.nn.functional as F
import torch
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MHAtt(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super(MHAtt, self).__init__()

        self.linear_v = nn.Linear(hid_dim, hid_dim)
        self.linear_k = nn.Linear(hid_dim, hid_dim)
        self.linear_q = nn.Linear(hid_dim, hid_dim)
        self.linear_merge = nn.Linear(hid_dim, hid_dim)
        self.hid_dim = hid_dim
        self.dropout = dropout
        self.nhead = n_heads
        self.dropout = nn.Dropout(dropout)
        self.hidden_size_head = int(self.hid_dim / self.nhead)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.nhead,
            self.hidden_size_head
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.nhead,
            self.hidden_size_head
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.nhead,
            self.hidden_size_head
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)  # 1,8,1,64
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.hid_dim
        )  # 1,1,512

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)  # 64

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)  # 1,8,1,1

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


class CAtt(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super(CAtt, self).__init__()

        self.mhatt = MHAtt(hid_dim, n_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, x, y, y_mask=None):
        x = self.norm(x + self.dropout(
            self.mhatt(y, y, x, y_mask)
        ))
        return x


class SAtt(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super(SAtt, self).__init__()

        self.mhatt = MHAtt(hid_dim, n_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, x, mask=None):
        x = self.norm(x + self.dropout(
            self.mhatt(x, x, x, mask)
        ))
        return x


# double residual

class Datt(nn.Module):
    def __init__(self, dim, nhead, dropout):
        super(Datt, self).__init__()
        self.self_att = SAtt(dim, nhead, dropout)
        self.cross_att = CAtt(dim, nhead, dropout)
        self.drug_norm = nn.LayerNorm(dim)
        self.protein_norm = nn.LayerNorm(dim)

    def forward(self, drug_vector, protein_vector):
        drug_vector_res = drug_vector
        protein_vector_res = protein_vector
        drug_vector = self.self_att(drug_vector, None)
        protein_vector = self.self_att(protein_vector, None)

        # Add residual connection for self-attention
        drug_vector = self.drug_norm(drug_vector + drug_vector_res)
        protein_vector = self.protein_norm(protein_vector + protein_vector_res)
        #
        # # Cross-attention
        drug_vector_res = drug_vector
        protein_vector_res = protein_vector
        drug_covector = self.cross_att(drug_vector, protein_vector, None)
        protein_covector = self.cross_att(protein_vector, drug_vector, None)

        # Add residual connection for cross-attention
        drug_covector = self.drug_norm(drug_covector + drug_vector_res)
        protein_covector = self.protein_norm(protein_covector + protein_vector_res)

        return drug_covector, protein_covector

# one—time residual
# class Datt(nn.Module):
#     def __init__(self, dim, nhead, dropout):
#         super(Datt, self).__init__()
#         self.self_att = SAtt(dim, nhead, dropout)
#         self.cross_att = CAtt(dim, nhead, dropout)
#         self.drug_norm = nn.LayerNorm(dim)
#         self.protein_norm = nn.LayerNorm(dim)
#
#     def forward(self, drug_vector, protein_vector):
#         drug_vector = self.self_att(drug_vector, None)
#         protein_vector = self.self_att(protein_vector, None)
#         # Cross-attention
#         drug_covector = self.cross_att(drug_vector, protein_vector, None)
#         protein_covector = self.cross_att(protein_vector, drug_vector, None)
#         return drug_covector, protein_covector

# triple
# class Datt(nn.Module):
#     def __init__(self, dim, nhead, dropout):
#         super(Datt, self).__init__()
#         self.self_att = SAtt(dim, nhead, dropout)
#         self.cross_att = CAtt(dim, nhead, dropout)
#         self.drug_norm = nn.LayerNorm(dim)
#         self.protein_norm = nn.LayerNorm(dim)
#
#     def forward(self, drug_vector, protein_vector):
#         drug_vector_res = drug_vector
#         protein_vector_res = protein_vector
#         drug_vector = self.self_att(drug_vector, None)
#         protein_vector = self.self_att(protein_vector, None)
#
#         # Add residual connection for self-attention
#         drug_vector = self.drug_norm(drug_vector + drug_vector_res*2)
#         protein_vector = self.protein_norm(protein_vector + protein_vector_res*2)
#         #
#         # # Cross-attention
#         drug_vector_res = drug_vector
#         protein_vector_res = protein_vector
#         drug_covector = self.cross_att(drug_vector, protein_vector, None)
#         protein_covector = self.cross_att(protein_vector, drug_vector, None)
#
#         # Add residual connection for cross-attention
#         drug_covector = self.drug_norm(drug_covector + drug_vector_res*2)
#         protein_covector = self.protein_norm(protein_covector + protein_vector_res*2)
#
#         return drug_covector, protein_covector

# remove Satt
# class Datt(nn.Module):
#     def __init__(self, dim, nhead, dropout):
#         super(Datt, self).__init__()
#         self.self_att = SAtt(dim, nhead, dropout)
#         self.cross_att = CAtt(dim, nhead, dropout)
#         self.drug_norm = nn.LayerNorm(dim)
#         self.protein_norm = nn.LayerNorm(dim)
#
#     def forward(self, drug_vector, protein_vector):
#         drug_vector_res = drug_vector
#         protein_vector_res = protein_vector
#         drug_covector = self.cross_att(drug_vector, protein_vector, None)
#         protein_covector = self.cross_att(protein_vector, drug_vector, None)
#
#         # Add residual connection for cross-attention
#         drug_covector = self.drug_norm(drug_covector + drug_vector_res)
#         protein_covector = self.protein_norm(protein_covector + protein_vector_res)
#
#         return drug_covector, protein_covector
# remove Catt
# class Datt(nn.Module):
#     def __init__(self, dim, nhead, dropout):
#         super(Datt, self).__init__()
#         self.self_att = SAtt(dim, nhead, dropout)
#         self.cross_att = CAtt(dim, nhead, dropout)
#         self.drug_norm = nn.LayerNorm(dim)
#         self.protein_norm = nn.LayerNorm(dim)
#
#     def forward(self, drug_vector, protein_vector):
#         drug_vector_res = drug_vector
#         protein_vector_res = protein_vector
#         drug_vector = self.self_att(drug_vector, None)
#         protein_vector = self.self_att(protein_vector, None)
#
#         # Add residual connection for self-attention
#         drug_vector = self.drug_norm(drug_vector + drug_vector_res)
#         protein_vector = self.protein_norm(protein_vector + protein_vector_res)
#
#
#         return drug_vector, protein_vector

# remove both
# class Datt(nn.Module):
#     def __init__(self, dim, nhead, dropout):
#         super(Datt, self).__init__()
#         self.drug_norm = nn.LayerNorm(dim)
#         self.protein_norm = nn.LayerNorm(dim)
#
#     def forward(self, drug_vector, protein_vector):
#
#
#         return drug_vector, protein_vector
