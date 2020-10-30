from pathlib import Path
import pytorch_lightning as pl


def get_project_dir():
    """Get the path to the project home."""
    path = Path(__file__).parent.parent
    project_dir = path.parent
    return project_dir


class HardNegativeCallback(pl.Callback):
    def __init__(self, max_num_hard_negatives=10, hard_negative_update_interval=10000):
        super().__init__()
        self.max_num_hard_negatives = max_num_hard_negatives
        self.hard_negative_update_interval = hard_negative_update_interval

        self.last_update = 0

    def on_validation_end(self, trainer: pl.Trainer, pl_module):
        if trainer.global_step >= self.last_update + self.hard_negative_update_interval:
            if trainer.train_dataloader.dataset.num_hard_negatives < self.max_num_hard_negatives:
                trainer.train_dataloader.dataset.num_hard_negatives += 1
                self.last_update = trainer.global_step
