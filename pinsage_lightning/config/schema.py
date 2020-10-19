from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from typing import Any


@dataclass
class DatasetConfig:
    dataset_path: str = MISSING
    random_walk_length: int = 2
    random_walk_restart_prob: float = 0.1
    num_random_walks: int = 5
    n_layers: int = 2
    batch_size: int = 32
    num_workers: int = 0
    num_neighbors: int = 3
    num_hard_negatives: int = 0


@dataclass
class ModelConfig:
    n_layers: int = 2
    hidden_dims: int = 16
    lr: float = 3e-5


@dataclass
class Config:
    dataset: DatasetConfig
    model: ModelConfig
    trainer: Any = MISSING

    num_epochs: int = 1
    k: int = 10


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
