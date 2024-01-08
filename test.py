import torch
import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
from dataset_h5 import StructureDataset
from torch_geometric.data import DataLoader
from model import HGPSLModel

def evaluation(result_dict, prefix):

    df = pd.DataFrame(result_dict).T
    true = df['true_label'].tolist()
    pred = df['pred_label'].tolist()
    probability = df['Probability'].tolist()

    out_name = prefix + '.csv'
    df_out = df[['pred_label','Probability']]
    df_out.to_csv(out_name, index_label='seq_name', header=['class','probability'])

    confusion = metrics.confusion_matrix(true, pred)
    TN, FP, FN, TP = metrics.confusion_matrix(true, pred).ravel()

    acc = metrics.accuracy_score(true, pred)
    precision = metrics.precision_score(true, pred)
    recall = metrics.recall_score(true, pred)
    f1 = metrics.f1_score(true, pred)
    mcc = metrics.matthews_corrcoef(true, pred)
    # auc = metrics.roc_auc_score(true, pred)
    Sn = TP / (TP + FN)
    Sp = TN / (TN + FP)

    fpr, tpr, _ = metrics.roc_curve(true, probability)
    auc = metrics.auc(fpr, tpr)

    return [acc, auc, precision, recall, f1, mcc, Sn, Sp]

def parse_args():

    parser = argparse.ArgumentParser('testing')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in testing')
    parser.add_argument('--test_dataset', type=str, default=None, help='dataset of test data')
    parser.add_argument('--model', type=str, default=None, help='path of model')
    parser.add_argument('--prefix', type=str, default=None, help='prefix of output csv')

    return parser.parse_args()

def main():

    args = parse_args()

    test_dataset = StructureDataset(args.test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = HGPSLModel()
    parameters = torch.load(args.model, map_location=torch.device(device))
    model.load_state_dict(parameters)
    model.to(device)
    torch.set_grad_enabled(False)
    model.eval()

    results_dict = {}
    for data in test_loader:
        data = data.to(device)
        label_pred = model(data)
        pred = label_pred.argmax(dim=1).data.cpu().detach()
        Probability = np.exp(label_pred[:, 1].data.cpu().detach())
        label = data.y.data.cpu()
        id = data.id

        for i in range(len(id)):
            seq_name = id[i]
            results_dict[seq_name] = {}
            results_dict[seq_name]['true_label'] = int(label[i])
            results_dict[seq_name]['pred_label'] = int(pred[i])
            results_dict[seq_name]['Probability'] = float(Probability[i])

    acc, auc, precision, recall, f1, mcc, Sn, Sp = evaluation(results_dict, args.prefix)

    print('acc:', acc, '\nauc:', auc, '\nprecision:', precision,
          '\nrecall:', recall, '\nF1-score:', f1, '\nmcc:', mcc, '\nSn:', Sn, '\nSp:', Sp)

if __name__ == '__main__':
    main()
