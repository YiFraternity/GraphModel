import random

import networkx as nx
import numpy as np
from gensim.models import Word2Vec

from utils import node_nbrs_weight


class Node2Vec:
    def __init__(self, g, walk_depth=5, p=1, q=1, embedding_size=32, window_size=3, workers=3, hs=1):
        self.model = None

        self.graph = g
        self.walk_depth = 5
        self.p = p
        self.q = q

        self.walk_depth = walk_depth
        self.embedding_size = embedding_size  # word2vec参数
        self.window_size = window_size  # word2vec参数
        self.workers = workers  # word2vec参数
        self.hs = hs  # word2vec参数

    def node2vec_walk(self, start_node):
        walk_nodes = [start_node]
        while len(walk_nodes) < self.walk_depth:
            cur_node = walk_nodes[-1]
            cur_nbrs = sorted(self.graph[cur_node])
            if len(cur_nbrs) > 0:
                if len(walk_nodes) == 1:
                    next_node = self.first_sample(cur_node)
                else:
                    prev_node = walk_nodes[-2]
                    next_node = self.second_sample(cur_node, prev_node)
                walk_nodes.append(next_node)
            else:
                break
        return walk_nodes

    def first_sample(self, cur_node):
        nbrs_node_list, nbrs_weight_list = node_nbrs_weight(self.graph, cur_node)
        nbrs_weight_list = [x / sum(nbrs_weight_list) for x in nbrs_weight_list]
        return np.random.choice(nbrs_node_list, p=nbrs_weight_list)  # 这里可以换成alias采样

    def second_sample(self, cur_node, pre_node):
        nbrs_node_list, nbrs_weight_list = node_nbrs_weight(self.graph, cur_node)
        for i in range(len(nbrs_node_list)):
            cur_nbr = nbrs_node_list[i]
            if cur_nbr == pre_node:
                t = self.p
            elif self.graph.has_edge(cur_nbr, pre_node):
                t = 1.
            else:
                t = self.q
            nbrs_weight_list[i] = nbrs_weight_list[i] / t
        nbrs_weight_list = [x / sum(nbrs_weight_list) for x in nbrs_weight_list]
        return np.random.choice(nbrs_node_list, p=nbrs_weight_list)  # 这里可以换成alias采样

    def get_model(self, context):
        self.model = Word2Vec(sentences=context, vector_size=self.embedding_size,
                              min_count=1, workers=self.workers, window=self.window_size,
                              hs=self.hs)

    def embedding(self, node):
        return self.model.wv[node]

    def embeddings(self):
        embedding_dict = dict()
        for node in self.graph.nodes:
            embedding_dict[node] = self.embedding(node)
        return embedding_dict

    def sequences(self):
        sequences = []
        nodes = list(self.graph.nodes)
        for _ in range(self.workers):
            random.shuffle(nodes)
            for v in nodes:
                sequences.append(self.node2vec_walk(start_node=v))
        return sequences


if __name__ == '__main__':
    G = nx.Graph()
    edges = [('a', 'b', 1), ('a', 'c', 3), ('b', 'd', 5), ('c', 'e', 2)]
    G.add_weighted_edges_from(edges)
    nv = Node2Vec(G)
    nv.get_model(nv.sequences())
    print(nv.embeddings())
