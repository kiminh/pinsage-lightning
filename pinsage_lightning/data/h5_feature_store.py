import math

import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm


class H5FeatureStore:
    def __init__(self, path, create_new=False):
        self.path = path
        self.create_new = create_new  # TODO support this arg
        self.feature_size = None

    def featurize_and_store(self, df, feature_fn, chunk_size, id_key, batch=False):
        dataset_size = len(df)

        if batch:
            feat_size = len(feature_fn(df.iloc[0:1])[0])
        else:
            feat_size = len(feature_fn(df.iloc[0]))

        with h5py.File(self.path, "w") as f:
            features = f.create_dataset("feature", (dataset_size, feat_size))
            item_ids = f.create_dataset(
                "item_id", (dataset_size,), dtype=h5py.string_dtype(encoding="ascii")
            )

            start = 0
            dfs, num_chunks = split_df(df, chunk_size)
            for df_chunk in tqdm(dfs, total=num_chunks):
                chunk_size = len(df_chunk)

                if batch:
                    embeddings = feature_fn(df_chunk)
                else:
                    embeddings = embed_df(df_chunk, feature_fn)

                features[start: start + chunk_size] = embeddings
                item_ids[start: start + chunk_size] = df_chunk[id_key]
                start += chunk_size

            return {item_id.decode(): i for i, item_id in enumerate(item_ids[:])}, feat_size

    def store(self, features, ids=None):
        dataset_size = len(ids)

        feat_size = len(features[0])

        with h5py.File(self.path, "w") as f:
            features_dataset = f.create_dataset("feature", (dataset_size, feat_size))
            item_ids = f.create_dataset(
                "item_id", (dataset_size,), dtype=h5py.string_dtype(encoding="ascii")
            )
            features_dataset[:] = features

            if ids is not None:
                item_ids[:] = ids

    def get_features(self, indices=None):
        with h5py.File(self.path, "r") as f:
            if indices is None:
                return f["feature"][:]
            else:
                return f["feature"][indices]

    def apply_pca(self, n_components, store=False):
        X = self.get_features()

        pca = PCA(n_components=n_components)
        pca.fit(X)
        X = pca.transform(X)

        if store:
            # TODO store along with existing item_ids if available
            self.store(X)

        return X


def split_df(df: pd.DataFrame, chunk_size: int):
    num_chunks = math.ceil(len(df) / chunk_size)

    return np.array_split(df, num_chunks), num_chunks


def embed_df(df: pd.DataFrame, feature_fn):
    embeddings = {}

    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        embeddings[i] = _embed_df(row, feature_fn)

    return list(embeddings.values())


def _embed_df(row, feature_fn):
    return feature_fn(row)
