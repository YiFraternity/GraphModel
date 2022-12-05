import random

import networkx as nx
import numpy as np
from gensim.models import Word2Vec

from utils import node_nbrs_weight


class DeepWalk:
    def __init__(self, g, walk_depth=10, workers=3, embed_size=32, window_size=1, sg=1, hs=1):
        self.model = None
        self.graph = g

        self.walk_depth = walk_depth
        self.workers = workers
        self.embed_size = embed_size

        self.window_size = window_size
        self.sg = sg
        self.hs = hs

    def sequences(self):
        sequences = []
        nodes = list(self.graph.nodes)
        for _ in range(self.workers):
            random.shuffle(nodes)
            for v in nodes:
                sequences.append(self.random_walk(v))
        return sequences

    def get_model(self, context):
        self.model = Word2Vec(sentences=context, vector_size=self.embed_size, window=self.window_size,
                              min_count=1, workers=self.workers)

    def embedding(self, node):
        return self.model.wv[node]

    def embeddings(self):
        embedding_dict = dict()
        for node in self.graph.nodes:
            embedding_dict[node] = self.embedding(node)
        return embedding_dict

    def random_walk(self, start_node):
        walk_nodes = [start_node]
        while len(walk_nodes) < self.walk_depth:
            cur_node = walk_nodes[-1]
            cur_nbrs = sorted(self.graph[cur_node])
            if len(cur_nbrs) > 0:
                nbrs_node_list, nbrs_weight_list = node_nbrs_weight(self.graph, cur_node)
                nbrs_weight_list = [x / sum(nbrs_weight_list) for x in nbrs_weight_list]
                next_node = np.random.choice(nbrs_node_list, p=nbrs_weight_list)  # 这里可以换成alias采样
                walk_nodes.append(next_node)
            else:
                break
        return walk_nodes


if __name__ == '__main__':
    edges = [('a', 'b', 1), ('a', 'c', 3), ('b', 'd', 5), ('c', 'e', 2)]
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    dp = DeepWalk(G)
    dp.get_model(dp.sequences())
    print(dp.embeddings())
