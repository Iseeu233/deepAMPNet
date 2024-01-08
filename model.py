import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GCN, HGPSLPool
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool as gep, global_max_pool as gmp, global_add_pool as gap

class GCNModel(nn.Module):
    def __init__(self, num_features=6165, output_dim=2,
                 dropout=0.2, aa_num=21, hidden_dim=512, net=None):
        super(GCNModel, self).__init__()

        self.num_features = num_features
        self.dp_ratio = dropout
        self.net = net

        if num_features <= 100:
            self.lin1 = nn.Linear(num_features, hidden_dim)
            aa_num = 0
        else:
            self.lin1 = nn.Linear(num_features-aa_num, hidden_dim)
            self.lin2 = nn.Linear(aa_num, aa_num)

        if self.net == 'GCN':
            self.conv1 = GCNConv(hidden_dim + aa_num, hidden_dim + aa_num)
            self.conv2 = GCNConv(hidden_dim + aa_num, (hidden_dim + aa_num) * 2)
            self.conv3 = GCNConv((hidden_dim + aa_num) * 2, (hidden_dim + aa_num) * 4)
        elif self.net == 'GAT':
            self.conv1 = GCNConv(hidden_dim + aa_num, hidden_dim + aa_num)
            self.conv2 = GCNConv(hidden_dim + aa_num, (hidden_dim + aa_num) * 2)
            self.conv3 = GATConv((hidden_dim + aa_num) * 2, (hidden_dim + aa_num) * 4)

        self.fc1 = nn.Linear((hidden_dim + aa_num) * 4 * 2, 1024)
        self.fc2 = nn.Linear(1024, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=self.dp_ratio)
        self.bn = nn.BatchNorm1d(1024)

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.num_features <= 100:
            x = self.relu(self.lin1(x))
        else:
            x1 = self.relu(self.lin1(x[:, 21:]))
            x2 = self.relu(self.lin2(x[:, :21]))
            x = torch.cat((x2, x1), 1)

        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.conv3(x, edge_index)
        x = self.relu(x)

        x1 = gap(x, batch)
        x2 = gep(x, batch)
        x = torch.cat((x1, x2), 1)

        x = self.relu(self.bn(self.fc1(x)))
        x = self.dropout(x)

        x = F.softmax(self.fc2(x), dim=1)

        return x

class HGPSLModel(nn.Module):
    def __init__(self, num_features=6165, aa_num=21, hidden_dim=512, output_dim=2, pool_ratio=0.5,
                 dropout=0.1, sample=True, sparse=True, sl=True, lamb=1.0):
        super(HGPSLModel, self).__init__()

        self.num_features = num_features
        self.dp_ratio = dropout

        if num_features <= 100:
            self.lin1 = nn.Linear(num_features, hidden_dim)
            aa_num = 0
        else:
            self.lin1 = nn.Linear(num_features-aa_num, hidden_dim)
            self.lin2 = nn.Linear(aa_num, aa_num)

        self.conv1 = GCNConv(hidden_dim + aa_num, hidden_dim)
        self.conv2 = GCN(hidden_dim, hidden_dim)
        self.conv3 = GCN(hidden_dim, hidden_dim)

        self.pool1 = HGPSLPool(hidden_dim, pool_ratio, sample, sparse, sl, lamb)
        self.pool2 = HGPSLPool(hidden_dim, pool_ratio, sample, sparse, sl, lamb)

        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None
        if self.num_features <= 100:
            x = self.relu(self.lin1(x))
        else:
            x1 = self.relu(self.lin1(x[:, 21:]))
            x2 = self.relu(self.lin2(x[:, :21]))
            x = torch.cat((x2, x1), 1)

        x = self.relu(self.conv1(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gep(x, batch)], dim=1)

        x = self.relu(self.conv2(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gep(x, batch)], dim=1)

        x = self.relu(self.conv3(x, edge_index, edge_attr))
        x3 = torch.cat([gmp(x, batch), gep(x, batch)], dim=1)

        x = F.relu(x1) + F.relu(x2) + F.relu(x3)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dp_ratio, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dp_ratio, training=self.training)
        x = F.log_softmax(self.fc3(x), dim=1)

        return x
