import networkx as nx
import pandas as pd

from deepwalk import DeepWalk
from line import Line
from node2vec import Node2Vec


def read_csv(fname):
    df = pd.read_csv(fname, header=None)
    edges = [tuple(_) for _ in df.values]
    return edges


if __name__ == '__main__':
    G = nx.Graph()
    graph_edges = read_csv('./graph.csv')
    G.add_weighted_edges_from(graph_edges)

    print('-------DeepWalk Embedding--------')
    deepwalk = DeepWalk(G)
    deepwalk.get_model(deepwalk.sequences())
    print(deepwalk.embeddings())

    print('-------Node2Vec Embedding--------')
    node2vec = Node2Vec(G)
    node2vec.get_model(node2vec.sequences())
    print(node2vec.embeddings())

    print('---------Line Embedding----------')
    line = Line(G)
    print(line.embedding)
