import os

import torch
from torch.utils.data import random_split, Dataset
import torch_geometric.utils

from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos


class SpectreGraphDataset(Dataset):
    def __init__(self, data_file):
        """ This class can be used to load the comm20, sbm and planar datasets. """
        base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, 'data')
        filename = os.path.join(base_path, data_file)
        self.adjs, self.eigvals, self.eigvecs, self.n_nodes, self.max_eigval, self.min_eigval, self.same_sample, self.n_max = torch.load(
            filename)
        print(f'Dataset {filename} loaded from file')

    def __len__(self):
        return len(self.adjs)

    def __getitem__(self, idx):
        adj = self.adjs[idx]
        n = adj.shape[-1]
        X = torch.ones(n, 1, dtype=torch.float)
        y = torch.zeros([1, 0]).float()
        edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
        edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
        edge_attr[:, 1] = 1
        num_nodes = n * torch.ones(1, dtype=torch.long)
        data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
                                         y=y, idx=idx, n_nodes=num_nodes)
        return data


class Comm20Dataset(SpectreGraphDataset):
    def __init__(self):
        super().__init__('community_12_21_100.pt')


class SBMDataset(SpectreGraphDataset):
    def __init__(self):
        super().__init__('sbm_200.pt')


class PlanarDataset(SpectreGraphDataset):
    def __init__(self):
        super().__init__('planar_64_200.pt')


class SpectreGraphDataModule(AbstractDataModule):
    def __init__(self, cfg, n_graphs=200):
        super().__init__(cfg)
        self.n_graphs = n_graphs
        self.prepare_data()
        self.inner = self.train_dataloader()

    def __getitem__(self, item):
        return self.inner[item]

    def prepare_data(self, graphs):
        test_len = int(round(len(graphs) * 0.2))
        train_len = int(round((len(graphs) - test_len) * 0.8))
        val_len = len(graphs) - train_len - test_len
        print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')
        splits = random_split(graphs, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(1234))

        datasets = {'train': splits[0], 'val': splits[1], 'test': splits[2]}
        super().prepare_data(datasets)


class Comm20DataModule(SpectreGraphDataModule):
    def prepare_data(self):
        graphs = Comm20Dataset()
        return super().prepare_data(graphs)


class SBMDataModule(SpectreGraphDataModule):
    def prepare_data(self):
        graphs = SBMDataset()
        return super().prepare_data(graphs)


class PlanarDataModule(SpectreGraphDataModule):
    def prepare_data(self):
        graphs = PlanarDataset()
        return super().prepare_data(graphs)


class SpectreDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = 'nx_graphs'
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = torch.tensor([1])               # There are no node types
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)

