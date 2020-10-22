import fire
import torch
from pinsage_lightning.config import DatasetConfig
from pinsage_lightning.data.h5_embedding_store import save_precomputed_embeddings_to_store
from pinsage_lightning.data.lightning_module import PinSAGEDataModule
from pinsage_lightning.model.lightning_module import PinSAGELightningModule


def extract_features(ckpt_path, dataset_path, output_path=None, use_gpu=False):
    model = PinSAGELightningModule.load_from_checkpoint(ckpt_path)

    cfg = DatasetConfig(dataset_path=dataset_path)
    dm = PinSAGEDataModule(cfg)
    dataloader = dm.val_dataloader()

    if use_gpu:
        model.cuda()

    # item_ids = []
    features = []
    for blocks in dataloader:
        if use_gpu:
            for i in range(len(blocks)):
                blocks[i] = blocks[i].cuda()
        # item_ids.extend(ids.numpy())
        features.append(model.get_repr(blocks))

    features = torch.cat(features, 0)
    if output_path:
        save_precomputed_embeddings_to_store(output_path, features.numpy())


if __name__ == "__main__":
    fire.Fire(extract_features)
