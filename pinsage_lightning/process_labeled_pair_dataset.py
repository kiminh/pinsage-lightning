import os
import tempfile

import pandas as pd

import fire
from pinsage_lightning.data.builder import PandasGraphBuilder
from pinsage_lightning.data.h5_embedding_store import build_h5_embedding_store
from pinsage_lightning.data.lightning_module import PinSAGEDataConfig
from pinsage_lightning.data.utils import (build_train_graph,
                                          build_val_test_matrix,
                                          linear_normalize,
                                          train_test_split_by_time)
from transformers import AutoModel, AutoTokenizer


def process_nowplaying_rs_dataset(data_dir, dst_dataset_path, model_name_or_path):
    users, tracks, events, labeled_pairs = load_data()

    graph_builder = PandasGraphBuilder()
    graph_builder.add_entities(users, "user_id", "user")
    graph_builder.add_entities(tracks, "track_id", "track")
    graph_builder.add_binary_relations(events, "user_id", "track_id", "listened")
    graph_builder.add_binary_relations(events, "track_id", "user_id", "listened-by")

    g = graph_builder.build()

    feature_fn = get_feature_fn(model_name_or_path)

    embedding_file = tempfile.mkstemp(suffix=".h5")

    item_id_to_node_id = build_h5_embedding_store(tracks, feature_fn, embedding_file, "track_id")
    g.nodes["track"].data["id"] = tracks.track_id.apply(lambda x: item_id_to_node_id[x])

    pairs_file = tempfile.mkstemp(suffix=".jsonl")
    convert_item_id_pairs(labeled_pairs, item_id_to_node_id, pairs_file)

    train_indices, val_indices, test_indices = train_test_split_by_time(
        events, "created_at", "user_id"
    )
    train_g = build_train_graph(
        g, train_indices, "user", "track", "listened", "listened-by"
    )
    assert train_g.out_degrees(etype="listened").min() > 0
    val_matrix, test_matrix = build_val_test_matrix(
        g, val_indices, test_indices, "user", "track", "listened"
    )

    dataset = PinSAGEDataConfig(train_g, "user", "track", embedding_file, pairs_file)
    dataset.save(dst_dataset_path)


def load_data(data_dir):
    data = pd.read_csv(os.path.join(data_dir, "context_content_features.csv"))
    track_feature_cols = list(data.columns[1:13])
    data = data[["user_id", "track_id", "created_at"] + track_feature_cols].dropna()

    users = data[["user_id"]].drop_duplicates()
    tracks = data[["track_id"] + track_feature_cols].drop_duplicates()
    assert tracks["track_id"].value_counts().max() == 1
    tracks = tracks.astype({"mode": "int64", "key": "int64", "artist_id": "category"})
    events = data[["user_id", "track_id", "created_at"]]
    events["created_at"] = (
        events["created_at"].values.astype("datetime64[s]").astype("int64")
    )

    return users, tracks, events, None  # TODO load pairs df


def get_feature_fn(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoModel.from_pretrained(model_name_or_path)

    def featurize(example):
        return model(**tokenizer(example["line"], return_tensors="pt"))[0][0].numpy()

    return featurize


def convert_item_id_pairs(labeled_pairs: pd.DataFrame, item_id_to_node_id, pairs_file, key_1="id_1", key_2="id_2"):
    labeled_pairs["node_id_1"] = labeled_pairs[key_1].apply(lambda x: item_id_to_node_id[x])
    labeled_pairs["node_id_2"] = labeled_pairs[key_2].apply(lambda x: item_id_to_node_id[x])

    converted_pairs = labeled_pairs[["node_id_1", "node_id_2"]]
    converted_pairs.to_json(pairs_file, orient="records", lines=True)


if __name__ == "__main__":
    fire.Fire(process_nowplaying_rs_dataset)
