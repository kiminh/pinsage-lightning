import os
import pickle

import pytorch_lightning as pl
import torch
import torchtext
from torch.utils.data import DataLoader

from pinsage_lightning.config import DatasetConfig
from pinsage_lightning.data.sampler import (
    ItemToItemBatchSampler,
    NeighborSampler,
    PinSAGECollator,
)
from pinsage_lightning.utils import get_project_dir


class PinSAGEDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DatasetConfig):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        cfg = self.cfg

        with open(os.path.join(get_project_dir(), cfg.dataset_path), "rb") as f:
            dataset = pickle.load(f)

            g = dataset["train-graph"]
            # val_matrix = dataset['val-matrix'].tocsr()
            # test_matrix = dataset['test-matrix'].tocsr()
            item_texts = dataset["item-texts"]
            user_ntype = dataset["user-type"]
            item_ntype = dataset["item-type"]
            # user_to_item_etype = dataset['user-to-item-type']
            # timestamp = dataset['timestamp-edge-column']

            # Assign user and movie IDs and use them as features (to learn an individual trainable
            # embedding for each entity)
            g.nodes[user_ntype].data["id"] = torch.arange(g.number_of_nodes(user_ntype))
            g.nodes[item_ntype].data["id"] = torch.arange(g.number_of_nodes(item_ntype))

        # Prepare torchtext dataset and vocabulary
        fields = {}
        examples = []
        for key, texts in item_texts.items():
            fields[key] = torchtext.data.Field(
                include_lengths=True, lower=True, batch_first=True
            )
        for i in range(g.number_of_nodes(item_ntype)):
            example = torchtext.data.Example.fromlist(
                [item_texts[key][i] for key in item_texts.keys()],
                [(key, fields[key]) for key in item_texts.keys()],
            )
            examples.append(example)
        textset = torchtext.data.Dataset(examples, fields)
        for key, field in fields.items():
            field.build_vocab(getattr(textset, key))
            # field.build_vocab(getattr(textset, key), vectors='fasttext.simple.300d')

        self.g = g
        self.user_ntype = user_ntype
        self.item_ntype = item_ntype
        self.textset = textset

    def train_dataloader(self):
        cfg = self.cfg

        # Sampler
        batch_sampler = ItemToItemBatchSampler(
            self.g, self.user_ntype, self.item_ntype, cfg.batch_size
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
            neighbor_sampler, self.g, self.item_ntype, self.textset
        )
        dataloader = DataLoader(
            batch_sampler,
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
