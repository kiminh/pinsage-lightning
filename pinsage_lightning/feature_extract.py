import fire
import torch
from pinsage_lightning.config import DatasetConfig
from pinsage_lightning.data.h5_feature_store import H5FeatureStore
from pinsage_lightning.data.lightning_module import PinSAGEDataModule
from pinsage_lightning.model.lightning_module import PinSAGELightningModule
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)


def extract_features(ckpt_path, dataset_path, output_path=None, use_gpu=False):
    logger.info(f"Loading model from {ckpt_path}")
    model = PinSAGELightningModule.load_from_checkpoint(ckpt_path)

    logger.info(f"Loading dataset from {dataset_path}")
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

        if len(blocks) % 10 == 0:
            print(len(blocks))

    features = torch.cat(features, 0)
    if output_path:
        logger.info("Saving features")
        store = H5FeatureStore(output_path)
        store.store(features.numpy())

    return features.numpy()


if __name__ == "__main__":
    fire.Fire(extract_features)
