import hydra
import pytorch_lightning as pl

from pinsage_lightning.config import Config, DatasetConfig, ModelConfig
from pinsage_lightning.data import PinSAGEDataModule
from pinsage_lightning.model import PinSAGELightningModule, PinSAGELightningModuleConfig


@hydra.main(config_path="config", config_name="config")
def train(cfg: Config):
    print(cfg.pretty())
    dm = get_data_module(cfg.dataset)

    model = get_model(dm.g, dm.item_ntype, dm.embedding_size, cfg.model)
    trainer = pl.Trainer(**cfg.trainer)

    trainer.fit(model, dm)


def get_data_module(cfg: DatasetConfig):
    dm = PinSAGEDataModule(cfg)
    dm.setup()
    return dm


def get_model(g, item_ntype, input_size: int, cfg: ModelConfig):
    config = PinSAGELightningModuleConfig(g, item_ntype, input_size, **cfg)
    model = PinSAGELightningModule(config)
    return model


if __name__ == "__main__":
    train()
