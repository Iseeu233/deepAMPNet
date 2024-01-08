import torch.nn as nn
import torch,os
from torch.nn.utils.rnn import PackedSequence
import numpy as np
from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).parent.parent
def pad_gap_scores(s, gap):
    col = gap.expand(s.size(0), 1)
    s = torch.cat([s, col], 1)
    row = gap.expand(1, s.size(1))
    s = torch.cat([s, row], 0)
    return s

class L2(nn.Module):
    def forward(self, x, y):
        return -torch.sum((x.unsqueeze(1)-y)**2, -1)

class L1(nn.Module):
    def forward(self, x, y):
        return -torch.sum(torch.abs(x.unsqueeze(1)-y), -1)

class OrdinalRegression(nn.Module):
    def __init__(self, embed_dim, n_classes, compare=L1()
                 , transform=None, align_method='ssa', beta_init=None
                 , allow_insertions=False, gap_init=-10
                 ):
        super(OrdinalRegression, self).__init__()

        self.n_in = embed_dim
        self.n_out = n_classes

        self.compare = compare
        self.align_method = align_method
        self.allow_insertions = allow_insertions
        self.gap = nn.Parameter(torch.FloatTensor([gap_init]))
        self.transform = transform

        if beta_init is None:
            # set beta to expectation of comparison
            # assuming embeddings are unit normal

            if type(compare) is L1:
                ex = 2 * np.sqrt(2 / np.pi) * embed_dim  # expectation for L1
                var = 4 * (1 - 2 / np.pi) * embed_dim  # variance for L1
            elif type(compare) is L2:
                ex = 4 * embed_dim  # expectation for L2
                var = 32 * embed_dim  # variance for L2
            else:
                ex = 0
                var = embed_dim

            beta_init = ex / np.sqrt(var)

        self.theta = nn.Parameter(torch.ones(1, n_classes - 1) / np.sqrt(var))
        self.beta = nn.Parameter(torch.zeros(n_classes - 1) + beta_init)

        self.clip()

    def clip(self):
        # clip the weights of ordinal regression to be non-negative
        self.theta.data.clamp_(min=0)

    def forward(self, z_x, z_y):
        return self.score(z_x, z_y)

    def score(self, z_x, z_y):

        s = self.compare(z_x, z_y)
        if self.allow_insertions:
            s = pad_gap_scores(s, self.gap)

        if self.align_method == 'ssa':
            a = torch.softmax(s, 1)
            b = torch.softmax(s, 0)

            if self.allow_insertions:
                index = s.size(0) - 1
                index = s.data.new(1).long().fill_(index)
                a = a.index_fill(0, index, 0)

                index = s.size(1) - 1
                index = s.data.new(1).long().fill_(index)
                b = b.index_fill(1, index, 0)

            a = a + b - a * b
            a = a / torch.sum(a)
        else:
            raise Exception('Unknown alignment method: ' + self.align_method)

        a = a.view(-1, 1)
        s = s.view(-1, 1)

        if hasattr(self, 'transform'):
            if self.transform is not None:
                s = self.transform(s)

        c = torch.sum(a * s)
        logits = c * self.theta + self.beta
        return logits.view(-1)

class BilinearContactMap(nn.Module):
    """
    Predicts contact maps as sigmoid(z_i W z_j + b)
    """
    def __init__(self, embed_dim, hidden_dim=1000, width=7, act=nn.LeakyReLU()):
        super(BilinearContactMap, self).__init__()

        self.scale = np.sqrt(hidden_dim)
        self.linear = nn.Linear(embed_dim, embed_dim) #, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))

    def clip(self):
        pass

    def forward(self, z):
        return self.predict(z)

    def predict(self, z):
        z_flat = z.view(-1, z.size(2))
        h = self.linear(z_flat).view(z.size(0), z.size(1), -1)
        s = torch.bmm(h, z.transpose(1,2))/self.scale + self.bias
        return s

class SkipLSTM(nn.Module):
    def __init__(self, nin, nout, hidden_dim, num_layers, dropout=0, bidirectional=True):
        super(SkipLSTM, self).__init__()

        self.nin = nin
        self.nout = nout

        self.dropout = nn.Dropout(p=dropout)

        self.layers = nn.ModuleList()
        dim = nin
        for i in range(num_layers):
            f = nn.LSTM(dim, hidden_dim, 1, batch_first=True, bidirectional=bidirectional)
            self.layers.append(f)
            if bidirectional:
                dim = 2*hidden_dim
            else:
                dim = hidden_dim

        n = hidden_dim*num_layers + nin
        if bidirectional:
            n = 2*hidden_dim*num_layers + nin

        self.proj = nn.Linear(n, nout)

    @staticmethod
    def load_pretrained(path='prose_dlm'):
        if path is None or path == 'prose_dlm':
            root = get_project_root()
            path = os.path.join(root, 'saved_models', 'prose_dlm_3x1024.sav')

        model = SkipLSTM(21, 21, 1024, 3)
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        return model

    def to_one_hot(self, x):
        packed = type(x) is PackedSequence
        if packed:
            one_hot = x.data.new(x.data.size(0), self.nin).float().zero_()
            one_hot.scatter_(1, x.data.unsqueeze(1), 1)
            one_hot = PackedSequence(one_hot, x.batch_sizes)
        else:
            one_hot = x.new(x.size(0), x.size(1), self.nin).float().zero_()
            one_hot.scatter_(2, x.unsqueeze(2), 1)
        return one_hot

    def transform(self, x):
        one_hot = self.to_one_hot(x)
        hs =  [one_hot] # []
        h_ = one_hot
        for f in self.layers:
            h,_ = f(h_)
            hs.append(h)
            h_ = h
        if type(x) is PackedSequence:
            h = torch.cat([z.data for z in hs], 1)
            h = PackedSequence(h, x.batch_sizes)
        else:
            h = torch.cat([z for z in hs], 2)
        return h

    def forward(self, x):
        one_hot = self.to_one_hot(x)
        hs = [one_hot]
        h_ = one_hot

        for f in self.layers:
            h,_ = f(h_)
            hs.append(h)
            h_ = h

        if type(x) is PackedSequence:
            h = torch.cat([z.data for z in hs], 1)
            z = self.proj(h)
            z = PackedSequence(z, x.batch_sizes)
        else:
            h = torch.cat([z for z in hs], 2)
            z = self.proj(h.view(-1,h.size(2)))
            z = z.view(x.size(0), x.size(1), -1)

        return z

class ProSEMT(nn.Module):
    def __init__(self, embedding, scop_predict, cmap_predict):
        super(ProSEMT, self).__init__()
        self.embedding = embedding
        self.scop_predict = scop_predict
        self.cmap_predict = cmap_predict

    @staticmethod
    def load_pretrained(path='prose_mt'):
        if path is None or path == 'prose_mt':
            root = get_project_root()
            path = os.path.join(root, 'deepAMPNet/Bi_LSTM_model', 'prose_mt_3x1024.sav')

        encoder = SkipLSTM(21, 21, 1024, 3)
        encoder.cloze = encoder.proj

        proj_in = encoder.proj.in_features
        proj = nn.Linear(proj_in, 100)
        encoder.proj = proj
        encoder.nout = 100

        scop_predict = OrdinalRegression(100, 5, compare=L1(), allow_insertions=False)
        cmap_predict = BilinearContactMap(proj_in)
        model = ProSEMT(encoder, scop_predict, cmap_predict)

        state_dict = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        return model

    def clip(self):
        self.scop_predict.clip()
        self.cmap_predict.clip()

    def forward(self, x):
        return self.embedding(x)

    def transform(self, x):
        return self.embedding.transform(x)

    def to_one_hot(self, x):
        return self.embedding.to_one_hot(x)

    def score(self, z_x, z_y):
        return self.scop_predict(z_x, z_y)

    def predict(self, z):
        return self.cmap_predict(z)
