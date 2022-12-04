import random

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


class _Line:
    def __init__(self, g, embed_size=32, seed=42, negative_ratio=5, order=2, lr=0.001, batch_size=32):
        self.debug_info = None
        self.edge_alias = None
        self.edge_prob = None
        self.sampling_table = None
        self.batch_size = None
        self.edges = None
        self.optimizer = None
        self._embeddings = None  # 节点本身的向量
        self.context_embeddings = None  # 节点作为其他节点的context节点时的向量
        self.node_size = None

        self.table_size = int(100)

        self.negative_ratio = negative_ratio
        self.graph = g
        self.embed_size = embed_size

        self.seed = seed

        if order == 1:
            self.loss = self.first_loss
        else:
            self.loss = self.second_loss

        self.build(lr=lr, batch_size=batch_size)

    def first_loss(self, sign, node1, node2):
        return -F.logsigmoid(sign * (self._embeddings[node1] * self._embeddings[node2]).sum(1)).mean()

    def second_loss(self, sign, node1, node2):
        return -F.logsigmoid(sign * (self._embeddings[node1] * self.context_embeddings[node2]).sum(1)).mean()

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if torch.cuda.is_available:
            torch.cuda.manual_seed(self.seed)

    def _look_up_dict(self):
        look_up_dict = dict()
        for i, node in enumerate(self.graph.nodes):
            look_up_dict[node] = i
        return look_up_dict

    def build(self, lr, batch_size):
        self.set_seed()

        self.node_size = len(self.graph.nodes)
        self._embeddings = nn.Parameter(nn.init.xavier_normal(torch.zeros(self.node_size, self.embed_size)),
                                        requires_grad=True)
        self.context_embeddings = nn.Parameter(nn.init.xavier_normal_(torch.zeros(self.node_size, self.embed_size)),
                                               requires_grad=True)
        self.optimizer = torch.optim.Adam([self._embeddings, self.context_embeddings], lr=lr)
        look_up_dict = self._look_up_dict()
        self.edges = [(look_up_dict[x[0]], look_up_dict[x[1]]) for x in self.graph.edges]
        self.batch_size = batch_size
        self.gen_sampling_table(look_up_dict)

    def gen_sampling_table(self, look_up_dict):
        node_degree = torch.zeros(self.node_size)
        power = 0.75

        for cur_node in self.graph.nodes:
            node_neighbors = self.graph[cur_node]
            node_weight_list = list()
            for k, v in node_neighbors.items():
                if 'weight' in v.keys():
                    weight = v['weight']
                else:
                    weight = 1
                node_weight_list.append(weight)
            node_degree[look_up_dict[cur_node]] = sum(node_weight_list)

        norm = float((node_degree ** power).sum())  # 度的3/4次方

        # 依节点度的3/4次方的概率进行采样
        self.sampling_table = np.zeros(self.table_size, dtype=np.int32)
        i, p = 0, 0.0
        for j in range(self.node_size):
            p += (node_degree[j] ** power) / norm
            while i < self.table_size and i / self.table_size < p:
                self.sampling_table[i] = j
                i += 1

        edge_num = len(self.graph.edges)

        temp_edge = list(self.graph.edges())[0]
        if 'weight' in self.graph[temp_edge[0]][temp_edge[1]]:
            total_sum = sum([self.graph[edge[0]][edge[1]]['weight'] for edge in self.graph.edges()])
            norm_prob = [self.graph[edge[0]][edge[1]]['weight'] * edge_num / total_sum for edge in self.graph.edges()]
        else:
            norm_prob = [1] * edge_num

        # 分解权值，将权值为w的边分解为w条边，这样每条边的权值都为1，不存在梯度跳跃问题
        # 边采样，使用Alias采样方法
        # 按照每条边所对应的的概率对边进行采样，这样能够有效的避免分解权值造成的存储问题，并且还能使目标函数仍然保持不变。
        small_weight_edge = np.zeros(edge_num, dtype=np.int32)
        large_weight_edge = np.zeros(edge_num, dtype=np.int32)
        small_wei_edge_num, large_wei_edge_num = 0, 0
        self.edge_prob = np.zeros(edge_num, dtype=np.float32)
        self.edge_alias = np.zeros(edge_num, dtype=np.int)  # alias 采样
        for k in range(edge_num - 1, -1, -1):
            if norm_prob[k] < 1:
                small_weight_edge[small_wei_edge_num] = k
                small_wei_edge_num += 1
            else:
                large_weight_edge[large_wei_edge_num] = k
                large_wei_edge_num += 1
        while small_wei_edge_num and large_wei_edge_num:  # alias 采样
            small_wei_edge_num -= 1
            cur_small_wei_edge = small_weight_edge[small_wei_edge_num]
            large_wei_edge_num -= 1
            cur_large_wei_edge = large_weight_edge[large_wei_edge_num]

            self.edge_prob[cur_small_wei_edge] = norm_prob[cur_small_wei_edge]
            self.edge_alias[cur_small_wei_edge] = cur_large_wei_edge
            norm_prob[cur_large_wei_edge] = norm_prob[cur_large_wei_edge] + norm_prob[cur_small_wei_edge] - 1
            if norm_prob[cur_large_wei_edge] < 1:  # alias 采样，若大的概率-（1-小概率）< 1，则将剩余部分当作小概率边
                small_weight_edge[small_wei_edge_num] = cur_large_wei_edge
                small_wei_edge_num += 1
            else:
                large_weight_edge[large_wei_edge_num] = cur_large_wei_edge
                large_wei_edge_num += 1

        while large_wei_edge_num:
            large_wei_edge_num -= 1
            self.edge_prob[large_weight_edge[large_wei_edge_num]] = 1
        while small_wei_edge_num:
            small_wei_edge_num -= 1
            self.edge_prob[small_weight_edge[small_wei_edge_num]] = 1

    def train_model(self):
        sum_loss = 0.0
        batches = self.batch_iter()
        batch_id = 0
        for batch in batches:
            h, t, sign = batch
            self.optimizer.zero_grad()
            cur_loss = self.loss(sign, h, t)
            sum_loss += cur_loss
            cur_loss.backward()
            self.optimizer.step()
            batch_id += 1  # feed用的参数表
        self.debug_info = sum_loss

    def batch_iter(self):
        table_size = self.table_size
        edges_num = len(self.graph.edges())
        shuffle_indices = torch.randperm(edges_num)
        sample_tag = 0  # 当tag=0是采样正样本，当tag=1时采样负样本
        sample_tag_size = 1 + self.negative_ratio

        h, t = [], []
        start_index = 0
        end_index = min(start_index + self.batch_size, edges_num)
        while start_index < edges_num:  # 采样边
            if sample_tag == 0:  # 正样本采样
                sign = 1
                h, t = [], []  # 每一轮存放一个batch
                for i in range(start_index, end_index):
                    if random.random() > self.edge_prob[shuffle_indices[i]]:  # 进行alias采样
                        shuffle_indices[i] = self.edge_alias[shuffle_indices[i]]
                    cur_h = self.edges[shuffle_indices[i]][0]
                    cur_t = self.edges[shuffle_indices[i]][1]
                    h.append(cur_h)
                    t.append(cur_t)
            else:  # 负样本采样
                sign = -1
                t = []
                for i in range(len(h)):
                    t.append(self.sampling_table[random.randint(0, table_size - 1)])
            yield h, t, torch.tensor([sign])
            sample_tag += 1
            sample_tag = sample_tag % sample_tag_size
            if sample_tag == 0:
                start_index = end_index
                end_index = min(start_index + self.batch_size, edges_num)

    def get_embedding(self):
        return self._embeddings.detach()


class Line:
    def __init__(self, g, embed_size=64, seed=42, negative_ratio=5):
        self.embedding = None
        self.model1 = _Line(g, embed_size=embed_size//2, seed=seed, negative_ratio=negative_ratio, order=1)
        self.model2 = _Line(g, embed_size=embed_size//2, seed=seed, negative_ratio=negative_ratio, order=2)
        for _ in range(10):
            self.model1.train_model()
            self.model2.train_model()
        self.embeddings()

    def embeddings(self):
        embedding1 = self.model1.get_embedding()
        embedding2 = self.model2.get_embedding()
        self.embedding = torch.cat((embedding1, embedding2), dim=1).detach()

    def save_embedding(self, fname):
        torch.save(self.embedding, fname)


if __name__ == '__main__':
    G = nx.Graph()
    edges = [('v1', 'v5'), ('v2', 'v5'), ('v3', 'v5'), ('v4', 'v5'),
             ('v1', 'v6'), ('v2', 'v6'), ('v3', 'v6'), ('v4', 'v6'),
             ('v6', 'v7'), ('v7', 'v8'), ('v7', 'v9'), ('v7', 'v10'),
             ]
    G.add_edges_from(edges)
    line = Line(G)
    line.save_embedding('./line_embed.pt')

    y = torch.load("./line_embed.pt")
    print(y)
