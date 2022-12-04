import networkx as nx


def node_nbrs_weight(graph:nx.Graph, node):
    nbrs_node_list, nbrs_weight_list = [], []
    for k, v in dict(graph[node]).items():
        if 'weight' in v.keys():
            weight = v['weight']
        else:
            weight = 1
        nbrs_node_list.append(k)
        nbrs_weight_list.append(weight)
    return nbrs_node_list, nbrs_weight_list

