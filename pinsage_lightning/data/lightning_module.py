import os
import pickle
from dataclasses import dataclass

import dgl
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from pinsage_lightning.config import DatasetConfig
from pinsage_lightning.data.labeled_pair_dataset import LabeledPairDataset
from pinsage_lightning.data.sampler import NeighborSampler, PinSAGECollator
from pinsage_lightning.utils import get_project_dir


@dataclass
class PinSAGEDataset:
    g: dgl.DGLGraph
    user_ntype: str
    item_ntype: str
    pairs_file: str
    embedding_file: str

    def save(self, save_path):
        os.makedirs(save_path, exist_ok=True)

        embedding_filename = os.path.basename(self.embedding_file)
        os.rename(self.embedding_file, os.path.join(save_path, embedding_filename))
        self.embedding_file = embedding_filename

        pairs_filename = os.path.basename(self.pairs_file)
        os.renmae(self.pairs_file, os.path.join(save_path, pairs_filename))
        self.pairs_file = pairs_filename

        pkl_path = os.path.join(save_path, "data.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            dataset = pickle.load(f)
        return dataset


class PinSAGEDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DatasetConfig):
        super().__init__()
        self.dataset = PinSAGEDataset.load(
            os.path.join(get_project_dir(), cfg.dataset_path)
        )
        self.cfg = cfg

        self.g = self.dataset.g
        self.user_ntype = self.dataset.user_ntype
        self.item_ntype = self.dataset.item_ntype
        self.embedding_file = self.dataset.embedding_file

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        cfg = self.cfg

        dataset = LabeledPairDataset(
            self.g, self.pairs_file, cfg.batch_size, cfg.num_hard_negatvies
        )

        neighbor_sampler = NeighborSampler(
            self.g,
            self.user_ntype,
            self.item_ntype,
            cfg.random_walk_length,
            cfg.random_walk_restart_prob,
            cfg.num_random_walks,
            cfg.num_neighbors,
            cfg.n_layers,
        )
        collator = PinSAGECollator(
            neighbor_sampler,
            self.g,
            self.item_ntype,
            self.textset,
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

        neighbor_sampler = NeighborSampler(
            self.g,
            self.user_ntype,
            self.item_ntype,
            cfg.random_walk_length,
            cfg.random_walk_restart_prob,
            cfg.num_random_walks,
            cfg.num_neighbors,
            cfg.n_layers,
        )

        collator = PinSAGECollator(
            neighbor_sampler, self.g, self.item_ntype, self.textset
        )
        dataloader_test = DataLoader(
            torch.arange(self.g.number_of_nodes(self.item_ntype)),
            batch_size=cfg.batch_size,
            collate_fn=collator.collate_test,
            num_workers=cfg.num_workers,
        )
        return dataloader_test
