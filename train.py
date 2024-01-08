import torch
import argparse
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn import metrics
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from dataset_h5 import StructureDataset
from torch_geometric.data import DataLoader

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
#torch.backends.cudnn.enabled = False

def evaluation(true, pred):
    confusion = metrics.confusion_matrix(true, pred)
    TN, FP, FN, TP = metrics.confusion_matrix(true, pred).ravel()

    acc = metrics.accuracy_score(true, pred)
    precision = metrics.precision_score(true, pred)
    recall = metrics.recall_score(true, pred)
    f1 = metrics.f1_score(true, pred)
    mcc = metrics.matthews_corrcoef(true, pred)
    auc = metrics.roc_auc_score(true, pred)
    Sn = TP / (TP + FN)
    Sp = TN / (TN + FP)

    return [acc, auc, precision, recall, f1, mcc, Sn, Sp]

def fit(model, optimizer, loss_func, scheduler, loader, train=True):
    if train:
        torch.set_grad_enabled(True)
        model.train()
    else:
        torch.set_grad_enabled(False)
        model.eval()
    running_loss = 0.0
    acc = 0.0
    true_list = []
    pred_list = []
    max_step = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for data in tqdm(loader, leave=False):
        data = data.to(device)
        max_step += 1
        if train:
            optimizer.zero_grad()
        label_pred = model(data)
        pred = label_pred.argmax(dim=1)
        label = data.y
        true_list.append(label.data.cpu())
        pred_list.append(pred.data.cpu().detach())
        # acc += (pred.data.cpu() == label.data.cpu()).sum()
        loss = loss_func(label_pred, label)
        running_loss += loss
        if train:
            loss.backward()
            optimizer.step()
        if max_step % 10 == 0:
            torch.cuda.empty_cache()
    true_list = np.concatenate(true_list)
    pred_list = np.concatenate(pred_list)
    running_loss = running_loss / (max_step)
    # avg_acc = acc / ((max_step) * batch_size)
    # if train:
    #     scheduler.step()
    return running_loss, true_list, pred_list

def train(model, optimizer, loss_func, scheduler, train_loader, val_loader, save_name, epochs):

    best_acc = 0
    best_auc = 0
    min_loss = 1000

    save_acc = save_name + '.best_acc.pth'
    save_auc = save_name + '.best_auc.pth'
    save_loss = save_name + '.min_loss.pth'
    save_log = save_name + '.log'
    log_file = open(save_log, 'w')

    train_loss_list = []
    val_loss_list = []
    train_accuracy_list = []
    val_accuracy_list = []

    for epoch in range(epochs):
        train_loss, train_true, train_pred = fit(model, optimizer, loss_func, scheduler, train_loader, train=True)
        val_loss, val_true, val_pred = fit(model, optimizer, loss_func, scheduler, val_loader, train=False)
        train_acc, train_auc = evaluation(train_true, train_pred)[:2]
        val_acc, val_auc = evaluation(val_true, val_pred)[:2]

        if val_acc > best_acc:
            best_acc = val_acc
            acc_metrics = evaluation(val_true, val_pred)
            torch.save(model.state_dict(), save_acc)
        if val_auc > best_auc:
            best_auc = val_auc
            auc_metrics = evaluation(val_true, val_pred)
            torch.save(model.state_dict(), save_auc)
        if val_loss < min_loss:
            min_loss = val_loss
            loss_metrics = evaluation(val_true, val_pred)
            torch.save(model.state_dict(), save_loss)

        train_loss_list.append(train_loss.cpu().detach())
        val_loss_list.append(val_loss.cpu().detach())
        train_accuracy_list.append(train_acc)
        val_accuracy_list.append(val_acc)

        print('Epoch', epoch + 1, '| train_loss: %.4f' % train_loss, '| train_acc:%.4f' % train_acc,
              '| val_loss: %.4f' % val_loss, '| val_acc:%.4f' % val_acc, file=log_file)
        print('Epoch', epoch + 1, '| train_loss: %.4f' % train_loss, '| train_acc:%.4f' % train_acc,
              '| val_loss: %.4f' % val_loss, '| val_acc:%.4f' % val_acc)

    save_metrics = save_name + '_metrics.tsv'
    df = pd.DataFrame({'best_acc': acc_metrics, 'best_auc': auc_metrics, 'min_loss': loss_metrics})
    df = df.T
    df.columns = ['acc', 'auc', 'precision', 'recall', 'f1', 'mcc', 'Sn', 'Sp']
    df = df.round(4)
    df = df.rename_axis('metrics')
    df.to_csv(save_metrics, index=True, header=True, sep='\t')

    drew(train_loss_list, train_accuracy_list, val_loss_list, val_accuracy_list, save_name, epochs)
    log_file.close()

def drew(train_loss_list, train_acc_list, test_loss_list, test_acc_list, png_name, epochs):
    plt.figure(figsize=(14, 7))
    plt.suptitle("AMP Train & Test Result")
    plt.subplot(121)
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(range(epochs), train_loss_list, label="train")
    plt.plot(range(epochs), test_loss_list, label="test")
    plt.grid(color="k", linestyle=":")
    plt.legend()
    plt.subplot(122)
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(ymax=1, ymin=0)
    plt.plot(range(epochs), train_acc_list, label="train")
    plt.plot(range(epochs), test_acc_list, label="test")
    plt.grid(color="k", linestyle=":")
    plt.legend()
    output = png_name + '.png'
    plt.savefig(output, dpi=600)
    plt.show()

def eval(model, val_loader, saved_model, device):

    parameters = torch.load(saved_model, map_location=torch.device(device))
    model.load_state_dict(parameters)
    model.to(device)
    torch.set_grad_enabled(False)
    model.eval()

    true_list = []
    pred_list = []
    for data in val_loader:
        data = data.to(device)
        label_pred = model(data)
        pred = label_pred.argmax(dim=1)
        label = data.y
        true_list.append(label.data.cpu())
        pred_list.append(pred.data.cpu().detach())

    true_list = np.concatenate(true_list)
    pred_list = np.concatenate(pred_list)
    eval_metrics = evaluation(true_list, pred_list)

    return eval_metrics

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--epochs', default=100, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--train_dataset', type=str, help='Path of the train dataset')
    parser.add_argument('--test_dataset', type=str, help='Path of the test dataset')
    parser.add_argument('--save', type=str, help='path of saving output results')
    parser.add_argument('--net', type=str, default='GCN',choices=['GCN', 'GAT', 'HGPSL'], help='GCN, GAT or HGPSL for model')

    parser.add_argument('--dropout', default=0.2, type=float, help='dropout ratio for model')
    parser.add_argument('--num_features', default=6165, type=int, help='number of input features')
    parser.add_argument('--hidden_dim', default=512, type=int, help='number of hidden dimensions')
    parser.add_argument('--aa_num', default=21, type=int, help='number of AA one-hot encoding')

    parser.add_argument('--sample_neighbor', type=bool, default=True, help='whether sample neighbors for HGPSL Model')
    parser.add_argument('--sparse_attention', type=bool, default=True, help='whether use sparse attention for HGPSL Model')
    parser.add_argument('--structure_learning', type=bool, default=True, help='whether perform structure learning for HGPSL Model')
    parser.add_argument('--pool_ratio', type=float, default=0.5, help='pooling ratio for HGPSL Model')
    parser.add_argument('--lamb', type=float, default=1.0, help='trade-off parameter for HGPSL Model')

    return parser.parse_args()

def main():

    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.net == 'HGPSL':
        from model import HGPSLModel
        model = HGPSLModel(num_features=args.num_features, aa_num=args.aa_num, hidden_dim=args.hidden_dim,
                           pool_ratio=args.pool_ratio, dropout=args.dropout, sample=args.sample_neighbor,
                           sparse=args.sparse_attention, sl=args.structure_learning, lamb=args.lamb)
        loss_func = nn.NLLLoss()
    else:
        from model import GCNModel
        model = GCNModel(num_features=args.num_features, aa_num=args.aa_num, hidden_dim=args.hidden_dim,
                         dropout=args.dropout, net=args.net)
        loss_func = nn.CrossEntropyLoss()
    model.to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    scheduler = False

    train_dataset = StructureDataset(args.train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_dataset = StructureDataset(args.test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    save_name = os.path.join(args.save, str(args.net))
    train(model, optimizer, loss_func, scheduler, train_loader, test_loader, save_name, args.epochs)

if __name__ == '__main__':
    main()
