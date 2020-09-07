import os
import pickle

import hydra
import pytorch_lightning as pl
import torch
import torchtext
from torch.utils.data import DataLoader

from pinsage_lightning.config import Config, DatasetConfig, ModelConfig
from pinsage_lightning.data.sampler import (ItemToItemBatchSampler,
                                            NeighborSampler, PinSAGECollator)
from pinsage_lightning.model import (PinSAGELightningModule,
                                     PinSAGELightningModuleConfig)
from pinsage_lightning.utils import get_project_dir


@hydra.main(config_path="config", config_name="config")
def train(cfg: Config):
    print(cfg.pretty())
    train_loader, test_loader, g, item_ntype, textset = get_dataloaders(cfg.dataset)

    model = get_model(g, item_ntype, textset, cfg.model)

    trainer = pl.Trainer(**cfg.trainer)

    trainer.fit(model, train_loader, test_loader)


def get_dataloaders(cfg: DatasetConfig):
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

    # Sampler
    batch_sampler = ItemToItemBatchSampler(g, user_ntype, item_ntype, cfg.batch_size)
    neighbor_sampler = NeighborSampler(
        g,
        user_ntype,
        item_ntype,
        cfg.random_walk_length,
        cfg.random_walk_restart_prob,
        cfg.num_random_walks,
        cfg.num_neighbors,
        cfg.n_layers,
    )
    collator = PinSAGECollator(neighbor_sampler, g, item_ntype, textset)
    dataloader = DataLoader(
        batch_sampler, collate_fn=collator.collate_train, num_workers=cfg.num_workers
    )
    dataloader_test = DataLoader(
        torch.arange(g.number_of_nodes(item_ntype)),
        batch_size=cfg.batch_size,
        collate_fn=collator.collate_test,
        num_workers=cfg.num_workers,
    )
    return dataloader, dataloader_test, g, item_ntype, textset


def get_model(g, item_ntype, textset: torchtext.data.Dataset, cfg: ModelConfig):
    config = PinSAGELightningModuleConfig(g, item_ntype, textset, **cfg)
    model = PinSAGELightningModule(config)
    return model


if __name__ == "__main__":
    train()
