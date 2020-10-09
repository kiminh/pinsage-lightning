import math

import h5py
import numpy as np
import pandas as pd
# from deco import concurrent, synchronized
from tqdm import tqdm


def build_h5_embedding_store(
    df: pd.DataFrame, feature_fn, filename: str, id_key: str, chunk_size: int = 1024, batch=False
):

    dataset_size = len(df)

    if batch:
        feat_size = len(feature_fn(df.iloc[0:1])[0])
    else:
        feat_size = len(feature_fn(df.iloc[0]))

    with h5py.File(filename, "w") as f:
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


def split_df(df: pd.DataFrame, chunk_size: int):
    num_chunks = math.ceil(len(df) / chunk_size)

    return np.array_split(df, num_chunks), num_chunks


# @synchronized
def embed_df(df: pd.DataFrame, feature_fn):
    embeddings = {}

    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        embeddings[i] = _embed_df(row, feature_fn)

    return list(embeddings.values())


# @concurrent(64)
def _embed_df(row, feature_fn):
    return feature_fn(row)
