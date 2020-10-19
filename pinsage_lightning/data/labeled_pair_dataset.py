import json
import random

import dgl
import torch
from torch.utils.data import IterableDataset


class LabeledPairDataset(IterableDataset):
    def __init__(
        self,
        g,
        filename,
        batch_size,
        num_hard_negatives,
        etype,
        etype_rev,
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
            etype_rev,
            etype,
        ] * hard_negative_distance

        self.negative_ids = self.get_negative_samples()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            lines_to_skip = 0
        else:
            lines_to_skip = worker_info.num_workers - 1

        while True:
            with open(self.filename) as f:
                if lines_to_skip > 0 and worker_info.id > 0:
                    for _ in range(worker_info.id):
                        f.readline()

                queries = []
                items = []

                for _ in range(self.batch_size):
                    line = json.loads(f.readline())
                    if lines_to_skip > 0:
                        for _ in range(lines_to_skip):
                            f.readline()
                    pos1, pos2 = line["node_id_1"], line["node_id_2"]
                    queries.append(pos1)
                    items.append(pos2)

                queries = torch.tensor(queries, dtype=torch.long)
                items = torch.tensor(items, dtype=torch.long)

                random.shuffle(self.negative_ids)
                neg = torch.LongTensor(self.negative_ids[: self.batch_size])

                if self.num_hard_negatives > 0:
                    hard_neg = dgl.sampling.random_walk(
                        self.g,
                        queries,
                        metapath=self.hard_negative_metapath,
                    )[0][:, -1]

                    indices = torch.randint(
                        0, self.batch_size, (self.num_hard_negatives,)
                    )
                    if torch.min(hard_neg[indices]) > -1:
                        neg[indices] = hard_neg[indices]

                yield queries, items, neg

    def get_negative_samples(self):
        negative_ids = set()
        with open(self.filename) as f:
            for line in f:
                line = json.loads(line)
                pos1, pos2 = line["node_id_1"], line["node_id_2"]
                negative_ids.add(pos1)
                negative_ids.add(pos2)
        negative_ids = list(negative_ids)
        random.shuffle(negative_ids)
        return negative_ids[: self.num_negative_samples]
