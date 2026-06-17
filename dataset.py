from pathlib import Path

from dgl.data.utils import load_graphs
import dgl
import torch
from utils import *

DATASET_CHOICES = ("questions", "reddit", "tolokers", "weibo", "yelp")
SPLIT_ID = 0
MISSING_RATIO = 0.995


def get_dataset_path(name, data_path):
    name = name.lower()
    if name not in DATASET_CHOICES:
        supported = ", ".join(DATASET_CHOICES)
        raise ValueError(f"Unsupported dataset: {name}. Supported: {supported}")

    path = Path(data_path) / name
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return path


class Dataset:
    def __init__(self, name="tolokers", data_path="data/all_data", del_ratio=0.005, seed=0):
        self.name = name.lower()

        graphs, _ = load_graphs(str(get_dataset_path(self.name, data_path)))
        if len(graphs) != 1:
            raise ValueError(f"Expected 1 graph for {self.name}, got {len(graphs)}")
        graph = graphs[0]

        for key in ("feature", "label", "train_masks", "val_masks", "test_masks"):
            if key not in graph.ndata:
                raise KeyError(f"Missing required node field: graph.ndata['{key}']")

        graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
        graph.ndata['feature'] = graph.ndata['feature'].float()
        graph.ndata['feature'], missing_mask = mask_features_with_zero(
            graph.ndata['feature'],
            missing_ratio=MISSING_RATIO,
            seed=seed,
        )
        graph.ndata['missing_mask'] = missing_mask
        graph.ndata['train_mask'] = graph.ndata['train_masks'][:, SPLIT_ID].bool()
        graph.ndata['val_mask'] = graph.ndata['val_masks'][:, SPLIT_ID].bool()
        graph.ndata['test_mask'] = graph.ndata['test_masks'][:, SPLIT_ID].bool()

        if del_ratio != 0:
            graph = random_delete(graph, del_ratio)
            graph = dgl.add_self_loop(graph)

        print(graph)

        self.graph = graph


def mask_features_with_zero(features, missing_ratio, seed):
    if not 0.0 <= missing_ratio <= 1.0:
        raise ValueError(f"missing_ratio must be in [0, 1], got {missing_ratio}")
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {tuple(features.shape)}")
    if not torch.is_floating_point(features):
        raise TypeError("features must be floating point")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    missing_mask = torch.rand(
        features.shape,
        generator=generator,
        device="cpu",
    ) < missing_ratio
    missing_mask = missing_mask.to(device=features.device)

    masked_features = features.clone()
    masked_features[missing_mask] = 0.0
    actual_ratio = missing_mask.sum().item() / missing_mask.numel()
    print('feature missing ratio: {:.3f}%'.format(actual_ratio * 100))
    return masked_features, missing_mask


def random_delete(graph, del_ratio):
    labels = graph.ndata['label']
    adj, edges, u, v = get_adj_from_edges(graph)
    sum = torch.sum(torch.concat((labels[u], labels[v]), dim=1), dim=1)
    index = torch.nonzero(sum == 1)
    he_edge_num = index.shape[0]
    threshold = int(del_ratio * he_edge_num)
    edge_to_move = index[torch.randperm(index.size(0))[:threshold]]
    graph_new = dgl.remove_edges(graph, list(edge_to_move))
    return graph_new


def train_delete(graph, train_mask, train_del):
    labels = graph.ndata['label']
    adj, edges, u, v = get_adj_from_edges(graph)
    false_indices = torch.where(train_mask == False)[0].to(graph.device)
    train_edge = torch.nonzero(torch.isin(v, false_indices))[:, 0]
    sum = torch.sum(torch.concat((labels[u], labels[v]), dim=1), dim=1)
    sum[train_edge] = 0
    index = torch.nonzero(sum == 1)
    he_edge_num = index.shape[0]
    threshold = int(train_del * he_edge_num)
    edge_to_move = index[torch.randperm(index.size(0))[:threshold]]
    graph_new = dgl.remove_edges(graph, list(edge_to_move))
    return graph_new

    
