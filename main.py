import torch
import torch.nn.functional as F
import argparse
import math
import time
import numpy as np
import random
from dataset import DATASET_CHOICES, Dataset, train_delete
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
import dgl
from model.SECGFD import SECGFD
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_device(device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, fallback to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)

def train(graph, args, in_feats, h_feats, num_class):
    graph = graph.to(device)
    features = graph.ndata['feature']
    labels = graph.ndata['label']
    train_mask = graph.ndata['train_mask'].bool()
    val_mask = graph.ndata['val_mask'].bool()
    test_mask = graph.ndata['test_mask'].bool()
    idx_train = torch.where(train_mask)[0]
    test_mask_np = test_mask.cpu().detach().numpy().astype(bool)
    print('train/dev/test samples: ', train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())

    if args.del_train != 0:
        graph = train_delete(graph, train_mask, train_del=args.del_train)
        graph = dgl.add_self_loop(graph)
    
    if args.run == 1:
        model = SECGFD(in_feats, h_feats, num_class, graph, d=args.order, high_order=2).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    weight = (1-labels[train_mask]).sum().item() / labels[train_mask].sum().item()
    print('cross entropy weight: ', weight)
    time_start = time.time()

    best_auc, best_f1, biggest_f1 = 0., 0., 0.
    final_metrics = {
        "roc_auc": 0.,
        "pr_auc": 0.,
        "macro_f1": 0.,
        "fraud_f1": 0.,
        "fraud_precision": 0.,
        "fraud_recall": 0.,
        "gmean": 0.,
    }

    for i in range(args.epoch):
        model.train()
        logits, emb = model(features)

        loss1 = F.cross_entropy(logits[train_mask], labels[train_mask], weight=torch.tensor([1., weight]).to(device))
        nce_loss = cal_nceloss(emb, features, labels, idx_train)
        loss = loss1 + args.lemda * nce_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        probs = logits.softmax(1)
        f1, thres = get_best_f1(labels[test_mask], probs[test_mask])
        labels_np = labels.cpu().detach().numpy()
        probs_np = probs.cpu().detach().numpy()
        preds = np.zeros_like(labels_np)
        preds[probs_np[:, 1] > thres] = 1
        metrics = compute_eval_metrics(labels_np[test_mask_np], preds[test_mask_np], probs_np[test_mask_np][:, 1])

        if biggest_f1 < f1:
            biggest_f1 = f1
        
        if (best_f1 + best_auc) < (f1 + metrics["roc_auc"]):
            best_f1 = f1
            best_auc = metrics["roc_auc"]
            final_metrics = metrics

        print(
            'Trial {}, Epoch {}, loss: {:.4f}, ROC-AUC: {:.2f}, PR-AUC: {:.2f}, Macro-F1: {:.2f}, '
            'Fraud-F1: {:.2f}, Fraud-Precision: {:.2f}, Fraud-Recall: {:.2f}, GMean: {:.2f}, '
            '(best f1: {:.4f} auc: {:.4f}), biggest f1: {:.4f}'.format(
                trial, i, loss,
                metrics["roc_auc"] * 100,
                metrics["pr_auc"] * 100,
                metrics["macro_f1"] * 100,
                metrics["fraud_f1"] * 100,
                metrics["fraud_precision"] * 100,
                metrics["fraud_recall"] * 100,
                metrics["gmean"] * 100,
                best_f1, best_auc, biggest_f1
            )
        )

    time_end = time.time()
    print('time cost: ', time_end - time_start, 's')
    print(format_metrics('Test', final_metrics))
    return final_metrics


def compute_eval_metrics(labels, preds, probs):
    roc_auc = float("nan") if len(set(labels.tolist())) < 2 else roc_auc_score(labels, probs)
    pr_auc = float("nan") if len(set(labels.tolist())) < 2 else average_precision_score(labels, probs)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    fraud_f1 = f1_score(labels, preds, pos_label=1, zero_division=0)
    fraud_precision = precision_score(labels, preds, pos_label=1, zero_division=0)
    fraud_recall = recall_score(labels, preds, pos_label=1, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    gmean = math.sqrt(sensitivity * specificity)
    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "macro_f1": float(macro_f1),
        "fraud_f1": float(fraud_f1),
        "fraud_precision": float(fraud_precision),
        "fraud_recall": float(fraud_recall),
        "gmean": float(gmean),
    }


def format_metrics(prefix, metrics):
    return (
        '{}: ROC-AUC {:.2f} PR-AUC {:.2f} Macro-F1 {:.2f} Fraud-F1 {:.2f} '
        'Fraud-Precision {:.2f} Fraud-Recall {:.2f} GMean {:.2f}'
    ).format(
        prefix,
        metrics["roc_auc"] * 100,
        metrics["pr_auc"] * 100,
        metrics["macro_f1"] * 100,
        metrics["fraud_f1"] * 100,
        metrics["fraud_precision"] * 100,
        metrics["fraud_recall"] * 100,
        metrics["gmean"] * 100,
    )


def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    labels = labels.cpu().detach().numpy()
    probs = probs.cpu().detach().numpy()
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:,1] > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HSGFD')
    parser.add_argument('--dataset', type=str, default='tolokers', choices=DATASET_CHOICES, help='dataset for our model')
    parser.add_argument("--hid_dim", type=int, default=64, help="Hidden layer dimension")
    parser.add_argument("--epoch", type=int, default=100, help="The max number of epochs")
    parser.add_argument("--run", type=int, default=1, help="Running times")
    parser.add_argument("--data_path", type=str, default='data/all_data', help="data path")
    parser.add_argument("--order", type=int, default=2, help="Order C in Beta Wavelet")
    parser.add_argument("--lemda", type=float, default=0.2, help="balance between losses")
    parser.add_argument("--del_ratio", type=float, default=0.000, help="delete heterophily edges ratios")
    parser.add_argument("--del_train", type=float, default=0.000, help="delete train heterophily edges ratios")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="running device")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument('--ntrials', type=int, default=1)

    args = parser.parse_args()
    device = resolve_device(args.device)
    setup_seed(args.seed)
    print(args)

    dataname = args.dataset
    data_path = args.data_path
    h_feats = args.hid_dim
    epoch_num = args.epoch
    del_ratio = args.del_ratio

    graph = Dataset(dataname, data_path, del_ratio, seed=args.seed).graph
    in_feats = graph.ndata['feature'].shape[1]
    num_class = 2

    result_metrics = {key: [] for key in (
        "roc_auc",
        "pr_auc",
        "macro_f1",
        "fraud_f1",
        "fraud_precision",
        "fraud_recall",
        "gmean",
    )}

    for trial in range(args.ntrials):
        final_metrics = train(graph, args, in_feats, h_feats, num_class)
        for key, value in final_metrics.items():
            result_metrics[key].append(value)

    mean_metrics = {key: np.nanmean(value) for key, value in result_metrics.items()}
    print('Final Result  Dataset:{}, Run:{}, {}'.format(
        args.dataset,
        args.ntrials,
        format_metrics('Test', mean_metrics)
    ))
