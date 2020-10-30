import os
import pickle
from dataclasses import dataclass

import dgl
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from pinsage_lightning.config import DatasetConfig
from pinsage_lightning.data.labeled_pair_dataset import LabeledPairDataset
from pinsage_lightning.data.sampler import NeighborSampler, PinSAGECollator
from torch.utils.data import DataLoader


@dataclass
class PinSAGEDataset:
    g: dgl.DGLHeteroGraph
    train_indices: np.ndarray
    user_ntype: str
    item_ntype: str
    etype: str
    etype_rev: str
    items_file: str = None
    embedding_size: int = None
    embedding_file: str = None
    pairs_file: str = None
    val_mask: np.ndarray = None
    test_mask: np.ndarray = None

    def save(self, save_path):
        os.makedirs(save_path, exist_ok=True)

        items_filename = os.path.basename(self.items_file)
        os.rename(self.items_file, os.path.join(save_path, items_filename))
        self.items_file = items_filename

        if self.embedding_file:
            embedding_filename = os.path.basename(self.embedding_file)
            os.rename(self.embedding_file, os.path.join(save_path, embedding_filename))
            self.embedding_file = embedding_filename

        pairs_filename = os.path.basename(self.pairs_file)
        os.rename(self.pairs_file, os.path.join(save_path, pairs_filename))
        self.pairs_file = pairs_filename

        pkl_path = os.path.join(save_path, "data.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(self, f)

        print(self.train_indices)
        print(self.g)

    @classmethod
    def load(cls, path):
        with open(os.path.join(path, "data.pkl"), "rb") as f:
            dataset = pickle.load(f)

        dataset.items_file = os.path.join(path, dataset.items_file)

        if dataset.embedding_file:
            dataset.embedding_file = os.path.join(path, dataset.embedding_file)
        dataset.pairs_file = os.path.join(path, dataset.pairs_file)

        return dataset

    def build_train_graph(self):
        return self.g.edge_subgraph(
            {self.etype: self.train_indices, self.etype_rev: self.train_indices},
            preserve_nodes=True,
        )


class PinSAGEDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DatasetConfig):
        super().__init__()
        self.dataset = PinSAGEDataset.load(
            hydra.utils.to_absolute_path(cfg.dataset_path)
        )
        self.cfg = cfg

        self.g = self.dataset.g
        self.user_ntype = self.dataset.user_ntype
        self.item_ntype = self.dataset.item_ntype
        self.embedding_size = self.dataset.embedding_size
        self.embedding_file = self.dataset.embedding_file
        self.pairs_file = self.dataset.pairs_file

        self.train_g = self.dataset.build_train_graph()

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        cfg = self.cfg

        dataset = LabeledPairDataset(
            self.train_g,
            self.pairs_file,
            cfg.batch_size,
            cfg.num_hard_negatives,
            self.dataset.etype,
            self.dataset.etype_rev,
        )

        neighbor_sampler = self.create_neighbor_sampler(train=True)
        collator = PinSAGECollator(
            neighbor_sampler,
            self.train_g,
            self.item_ntype,
            embedding_file=self.embedding_file,
        )
        dataloader = DataLoader(
            dataset,
            collate_fn=collator.collate_train,
            num_workers=cfg.num_workers,
        )

        return dataloader

    def val_dataloader(self):
        cfg = self.cfg

        neighbor_sampler = self.create_neighbor_sampler(train=False)

        collator = PinSAGECollator(
            neighbor_sampler,
            self.g,
            self.item_ntype,
            embedding_file=self.embedding_file,
        )
        dataloader_test = DataLoader(
            torch.arange(self.num_item_nodes),
            batch_size=cfg.batch_size,
            collate_fn=collator.collate_test,
            num_workers=cfg.num_workers,
        )
        return dataloader_test

    @property
    def num_item_nodes(self):
        return self.g.number_of_nodes(self.item_ntype)

    def create_neighbor_sampler(self, train=False):
        cfg = self.cfg

        if train:
            g = self.train_g
        else:
            g = self.g

        neighbor_sampler = NeighborSampler(
            g,
            self.user_ntype,
            self.item_ntype,
            cfg.random_walk_length,
            cfg.random_walk_restart_prob,
            cfg.num_random_walks,
            cfg.num_neighbors,
            cfg.n_layers,
        )

        return neighbor_sampler
