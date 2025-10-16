import numpy as np
import torch
from torch_geometric.nn import to_hetero, GraphConv
from torch.nn import Linear, Dropout, Sigmoid
import lightgbm as lgb
import random
import torch.nn.functional as F
import torch.nn as nn
import deep_attention
import math
class GNNEncoder(torch.nn.Module):

    def __init__(self, hidden_channels, out_channels, p=0.2):
        super().__init__()

        self.conv_in = GraphConv((-1, -1), hidden_channels, aggr='sum')
        self.conv_med = GraphConv((-1, -1), hidden_channels, aggr='sum')
        self.conv_out = GraphConv((-1, -1), out_channels, aggr='sum')
        self.sigmod = Sigmoid()
        self.dropout = Dropout(p)

    def forward(self, x, edge_index):
        # print("x is:" + x, "edge_index is" + edge_index)
        x = self.conv_in(x, edge_index)
        x = self.sigmod(x)
        x = self.dropout(x)

        for i in range(2):
            x = self.conv_med(x, edge_index)  # direcly ouput dimension
            # print(f'med layer {i+2}', x)
            x = self.sigmod(x)  # apply activation
            x = self.dropout(x)  # apply dropout

        x = self.conv_out(x, edge_index)  # direcly ouput dimension
        # print(f'final layer 4: ', x)
        # print(x)
        return x


class EdgeClassifier(torch.nn.Module):

    def __init__(self, hidden_channels):
        super().__init__()
        self.deep_inter_att=deep_attention.Datt(dim=64,nhead=4,dropout=0.2)
        self.lin1 = Linear(64*2, hidden_channels)
        self.bn1=nn.BatchNorm1d(hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.lin3 = Linear(hidden_channels, 1)

        self.drug_proj=nn.Linear(hidden_channels, 64)
        self.protein_proj=nn.Linear(hidden_channels, 64)

    # 这里的z_dict就是train_data.x_dict就是一个包含蛋白质和药物的字典

    def forward(self, z_dict, edge_label_index):

        row, col = edge_label_index

        # print(z_dict['drug'][row],z_dict['protein'][col])
        # z = torch.cat([z_dict['drug'][row], z_dict['protein'][col]], dim=-1)
        drug_embed=z_dict['drug'][row]
        protein_embed=z_dict['protein'][col]

        drug_embed=self.drug_proj(drug_embed).unsqueeze(1)
        protein_embed=self.protein_proj(protein_embed).unsqueeze(1)
        # print(drug_embed.shape)
        # print(protein_embed.shape)
        drug_convector,protein_convector=self.deep_inter_att(drug_embed,protein_embed)
        z=torch.cat([drug_convector.mean(dim=1),protein_convector.mean(dim=1)],dim=-1)
        # print(z)
        z = self.bn1(F.relu(self.lin1(z)))
        z = self.bn2(F.relu(self.lin2(z)))
        z = self.lin3(z)
        return z.view(-1)


class Model(torch.nn.Module):

    def __init__(self, hidden_channels, data):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = EdgeClassifier(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        out = self.decoder(z_dict, edge_label_index)
        return z_dict, out

class EarlyStopper():
    def __init__(self, tolerance=10, min_delta=0.05):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0

    def early_stop(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                return True
        return False


def shuffle_label_data(train_data, a=('drug', 'interaction', 'protein')):
    length = train_data[a].edge_label.shape[0]
    lff = list(np.arange(length))
    random.shuffle(lff)

    train_data[a].edge_label = train_data[a].edge_label[lff]
    train_data[a].edge_label_index = train_data[a].edge_label_index[:, lff]
    return train_data
