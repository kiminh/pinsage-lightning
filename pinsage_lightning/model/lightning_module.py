from dataclasses import dataclass

import dgl
import pytorch_lightning as pl
import torch
from torch.optim import Adam

from .layers import ItemToItemScorer, LinearProjector, SAGENet


@dataclass
class PinSAGELightningModuleConfig:
    full_graph: dgl.DGLGraph
    ntype: str
    input_size: int
    hidden_dims: int
    hidden_dims: int = 16
    n_layers: int = 2
    lr: float = 3e-5


class PinSAGELightningModule(pl.LightningModule):
    def __init__(self, cfg: PinSAGELightningModuleConfig):
        super().__init__()
        self.cfg = cfg

        self.proj = torch.nn.Linear(cfg.input_size, cfg.hidden_size)
        self.sage = SAGENet(cfg.hidden_dims, cfg.n_layers)
        self.scorer = ItemToItemScorer(cfg.full_graph, cfg.ntype)

    def forward(self, pos_graph, neg_graph, blocks):
        h_item = self.get_repr(blocks)
        pos_score = self.scorer(pos_graph, h_item)
        neg_score = self.scorer(neg_graph, h_item)
        return (neg_score - pos_score + 1).clamp(min=0)

    def get_repr(self, blocks):
        h_item = self.proj(blocks[0].srcdata["feature"])
        h_item_dst = self.proj(blocks[-1].dstdata["feature"])
        return h_item_dst + self.sage(blocks, h_item)

    def training_step(self, batch, batch_idx):
        pos_graph, neg_graph, blocks = batch
        loss = self.forward(pos_graph, neg_graph, blocks).mean()
        result = pl.TrainResult(loss)
        result.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return result

    def validation_step(self, *args, **kwargs) -> pl.EvalResult:
        pass

    def configure_optimizers(self):
        opt = Adam(self.parameters(), lr=self.cfg.lr)
        return opt
