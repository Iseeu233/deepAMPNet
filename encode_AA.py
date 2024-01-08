import os
import h5py
import torch
import argparse
import numpy as np
from pyfaidx import Fasta
from Bi_LSTM import ProSEMT

class Alphabet:
    def __init__(self, chars, encoding=None, mask=False, missing=255):
        self.chars = np.frombuffer(chars, dtype=np.uint8)
        self.encoding = np.zeros(256, dtype=np.uint8) + missing
        if encoding is None:
            self.encoding[self.chars] = np.arange(len(self.chars))
            self.size = len(self.chars)
        else:
            self.encoding[self.chars] = encoding
            self.size = encoding.max() + 1
        self.mask = mask
        if mask:
            self.size -= 1

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return chr(self.chars[i])

    def encode(self, x):
        """ encode a byte string into alphabet indices """
        x = np.frombuffer(x, dtype=np.uint8)
        return self.encoding[x]

    def decode(self, x):
        """ decode index array, x, to byte string of this alphabet """
        string = self.chars[x]
        return string.tobytes()

    def unpack(self, h, k):
        """ unpack integer h into array of this alphabet with length k """
        n = self.size
        kmer = np.zeros(k, dtype=np.uint8)
        for i in reversed(range(k)):
            c = h % n
            kmer[i] = c
            h = h // n
        return kmer

    def get_kmer(self, h, k):
        """ retrieve byte string of length k decoded from integer h """
        kmer = self.unpack(h, k)
        return self.decode(kmer)

class Uniprot21(Alphabet):
    def __init__(self, mask=False):
        chars = alphabet = b'ARNDCQEGHILKMFPSTWYVXOUBZ'
        encoding = np.arange(len(chars))
        encoding[21:] = [11,4,20,20] # encode 'OUBZ' as synonyms
        super(Uniprot21, self).__init__(chars, encoding=encoding, mask=mask, missing=20)

def embed_AA(model, x, num_features=6165, pool='none', use_cuda=False,device = None):

    if len(x) == 0:
        n = model.embedding.proj.weight.size(1)
        z = np.zeros((1,n), dtype=np.float32)
        return z

    alphabet = Uniprot21()
    x = x.upper()
    x = bytes(x, encoding='utf-8')
    x = alphabet.encode(x)
    x = torch.from_numpy(x)
    if use_cuda:
        x = x.to(device)

    # embed the sequence
    with torch.no_grad():
        x = x.long().unsqueeze(0)
        if num_features == 21:
            z = model.to_one_hot(x)
        elif num_features == 100:
            z = model(x)
        else:
            z = model.transform(x)[:, :, :num_features]

        z = z.squeeze(0)
        if pool == 'sum':
            z = z.sum(0)
        elif pool == 'max':
            z,_ = z.max(0)
        elif pool == 'avg':
            z = z.mean(0)

    return z

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input fasta file')
    parser.add_argument('-o', '--output', help='output h5 file')
    parser.add_argument('-n', '--num', type=int, choices=[21, 100, 2069, 4117, 6165],
                        help='''number of output features:
                             21: one-hot embedding of AA
                             100: embedding result of 3 Bi-LSTM layers
                             2069: 21+1024*2, 1 Bi-LSTM layer with one-hot
                             4117: 21+1024*4, 2 Bi-LSTM layers with one-hot
                             6165: 21+1024*6, 3 Bi-LSTM layers with one-hot''')
    parser.add_argument('--pool', choices=['none', 'sum', 'max', 'avg'], default='none',
                        help='apply some sort of pooling operation over each sequence (default: none)')
    parser.add_argument('-d', '--device', type=int, default=-2, help='compute device to use')
    args = parser.parse_args()

    model = ProSEMT.load_pretrained().eval()

    d = args.device
    use_cuda = (d != -1) and torch.cuda.is_available()
    if d >= 0:
        device = torch.device('cuda', d)
    else:
        device = torch.device('cpu')
    if use_cuda:
        model.to(device)

    h5 = h5py.File(args.output, 'w')

    pool = args.pool
    f = Fasta(args.input)
    for name, sequence in f.items():
        z = embed_AA(model=model, x=str(sequence), num_features=args.num,
                     pool=pool, use_cuda=use_cuda,device=device)
        z = z.cpu().numpy()
        h5.create_dataset(name, data=z)

if __name__ == "__main__":
    main()
