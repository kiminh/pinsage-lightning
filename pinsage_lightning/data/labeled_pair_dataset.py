import random

import dgl
import torch
from torch.utils.data import IterableDataset

import h5py


class LabeledPairDataset(IterableDataset):
    def __init__(
        self,
        g,
        filename,
        batch_size,
        num_hard_negatives,
        hard_negative_distance=1,
        num_negative_samples=1e5,
    ):
        self.g = g
        self.filename = filename
        self.batch_size = batch_size
        self.num_hard_negatives = num_hard_negatives

        self.num_negative_samples = int(num_negative_samples)
        self.hard_negative = False
        self.hard_negative_metapath = [
            self.item_to_user_etype,
            self.user_to_item_etype,
        ] * hard_negative_distance

        self.negative_ids = self.get_negative_samples()

    def __iter__(self):
        while True:
            with open(self.filename) as f:
                queries = []
                items = []

                for _ in range(self.batch_size):
                    line = f.readline()
                    pos1, pos2 = line["node_id_1"], line["node_id_2"]
                    queries.append(pos1)
                    items.append(pos2)

                queries = torch.tensor(queries, dtype=torch.int)
                items = torch.tensor(items, dtype=torch.int)

                neg = torch.tensor(random.shuffle(self.negative_ids)[: self.batch_size])

                if self.num_hard_negatives > 0:
                    hard_neg = dgl.sampling.random_walk(
                        self.g,
                        queries,
                        metapath=self.hard_negative_metapath,
                    )[0][:, -1]

                    indices = torch.randint(
                        0, self.batch_size + 1, (self.num_hard_negatives,)
                    )
                    neg[indices] = hard_neg

                yield queries, items, neg

    def get_negative_samples(self):
        negative_ids = set()
        with open(self.filename) as f:
            for line in f:
                pos1, pos2 = line["node_id_1"], line["node_id_2"]
                negative_ids.add(pos1)
                negative_ids.add(pos2)
        return random.shuffle(list(negative_ids))[: self.num_negative_samples]


class LabeledPairWithEmbeddingsDataset(IterableDataset):
    def __init__(self, dataset, embedding_file):
        self.dataset = dataset
        self.embedding_file = embedding_file

        self.h5 = None

    def __iter__(self):
        if not self.h5:
            self.h5 = h5py.File(self.embedding_file, "r")

        for batch in self.dataset:
            queries, items, neg = batch

            queries = self.h5["feature"][queries]
            items = self.h5["feature"][items]
            neg = self.h5["feature"][neg]
            yield queries, items, neg
