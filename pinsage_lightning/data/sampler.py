import dgl
import h5py
import torch


def compact_and_copy(frontier, seeds):
    block = dgl.to_block(frontier, seeds)
    for col, data in frontier.edata.items():
        if col == dgl.EID:
            continue
        block.edata[col] = data[block.edata[dgl.EID]]
    return block


class NeighborSampler(object):
    def __init__(
        self,
        g,
        user_type,
        item_type,
        random_walk_length,
        random_walk_restart_prob,
        num_random_walks,
        num_neighbors,
        num_layers,
    ):
        self.g = g
        self.user_type = user_type
        self.item_type = item_type
        self.user_to_item_etype = list(g.metagraph()[user_type][item_type])[0]
        self.item_to_user_etype = list(g.metagraph()[item_type][user_type])[0]
        self.samplers = [
            dgl.sampling.PinSAGESampler(
                g,
                item_type,
                user_type,
                random_walk_length,
                random_walk_restart_prob,
                num_random_walks,
                num_neighbors,
            )
            for _ in range(num_layers)
        ]

    def sample_blocks(self, seeds, heads=None, tails=None, neg_tails=None):
        blocks = []
        for sampler in self.samplers:
            frontier = sampler(seeds)
            if heads is not None:
                eids = frontier.edge_ids(
                    torch.cat([heads, heads]),
                    torch.cat([tails, neg_tails]),
                    return_uv=True,
                )[2]
                if len(eids) > 0:
                    old_frontier = frontier
                    frontier = dgl.remove_edges(old_frontier, eids)
                    # print(old_frontier)
                    # print(frontier)
                    # print(frontier.edata['weights'])
                    # frontier.edata['weights'] = old_frontier.edata['weights'][frontier.edata[dgl.EID]]
            block = compact_and_copy(frontier, seeds)
            seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return blocks

    def sample_from_item_pairs(self, heads, tails, neg_tails):
        # Create a graph with positive connections only and another graph with negative
        # connections only.
        pos_graph = dgl.graph(
            (heads, tails), num_nodes=self.g.number_of_nodes(self.item_type)
        )
        neg_graph = dgl.graph(
            (heads, neg_tails), num_nodes=self.g.number_of_nodes(self.item_type)
        )
        pos_graph, neg_graph = dgl.compact_graphs([pos_graph, neg_graph])
        seeds = pos_graph.ndata[dgl.NID]

        blocks = self.sample_blocks(seeds, heads, tails, neg_tails)
        return pos_graph, neg_graph, blocks


def assign_simple_node_features(ndata, g, ntype, assign_id=False):
    """
    Copies data to the given block from the corresponding nodes in the original graph.
    """
    for col in g.nodes[ntype].data.keys():
        if not assign_id and col == dgl.NID:
            continue
        induced_nodes = ndata[dgl.NID]
        ndata[col] = g.nodes[ntype].data[col][induced_nodes]


def assign_features_to_blocks(blocks, g, ntype):
    # For the first block (which is closest to the input), copy the features from
    # the original graph as well as the texts.
    assign_simple_node_features(blocks[0].srcdata, g, ntype)
    assign_simple_node_features(blocks[-1].dstdata, g, ntype)


def assign_embeddings_from_file_to_blocks(blocks, h5):
    blocks[0].srcdata["feature"] = torch.tensor(h5["feature"][blocks[0].srcdata["id"]])
    blocks[1].dstdata["feature"] = torch.tensor(h5["feature"][blocks[1].dstdata["id"]])


class PinSAGECollator(object):
    def __init__(self, sampler, g, ntype, embedding_file=None):
        self.sampler = sampler
        self.ntype = ntype
        self.g = g

        self.embedding_file = embedding_file
        self.embeddings = None

    def collate_train(self, batches):
        heads, tails, neg_tails = batches[0]
        # Construct multilayer neighborhood via PinSAGE...
        pos_graph, neg_graph, blocks = self.sampler.sample_from_item_pairs(
            heads, tails, neg_tails
        )
        assign_features_to_blocks(blocks, self.g, self.ntype)

        if self.embedding_file:
            if not self.embeddings:
                self.embeddings = h5py.File(self.embedding_file)

            assign_embeddings_from_file_to_blocks(blocks, self.embeddings)

        return pos_graph, neg_graph, blocks

    def collate_test(self, samples):
        batch = torch.LongTensor(samples)
        blocks = self.sampler.sample_blocks(batch)
        assign_features_to_blocks(blocks, self.g, self.ntype)

        if self.embedding_file:
            if not self.embeddings:
                self.embeddings = h5py.File(self.embedding_file)

            assign_embeddings_from_file_to_blocks(blocks, self.embeddings)

        return blocks
