import copy
import json
import numpy as np
from munkres import Munkres
from sklearn import metrics
import scipy.sparse as sp
import torch
import random
from torch_geometric.data import InMemoryDataset, download_url
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import normalize


def get_edge_index(data,adj_noeye):
    """
    使用官网提供的图神经网络时使用
    :param data:
    :return:
    """
    coo = []
    for d in data:
        temp_coo = []
        # edge_weights = []
        for i in range(len(d) - 1):
            for j in range(i + 1, len(d)):
                adj_value = adj_noeye[d[i]][d[j]]
                if adj_value != 0:
                    temp_coo.append([i, j])
                    temp_coo.append([i, j])
        coo_T = torch.tensor(np.transpose(np.array(temp_coo))).to(torch.int64)
        coo.append(coo_T)
        # edge_weights_array = torch.Tensor(np.array(edge_weights)).cuda()
    return coo

def get_data_label(data_path,word2ix,dis2idx,adj_noeye):
    data = []
    label = []
    with open(data_path, 'r', encoding='utf-8') as f:
        shuffle_data = json.load(f)

    for i in range(len(shuffle_data)):
        words_to_idx = []
        for sym in shuffle_data[i]['symptom']:
            words_to_idx.append(word2ix[sym])

        label_to_idx = [0]*6
        for lab in shuffle_data[i]['type']:
            label_to_idx[dis2idx[lab]] = 1

        data.append(torch.tensor(words_to_idx))
        label.append(torch.tensor(label_to_idx))
    # adj, M_matrix = self.get_adj(data)
    coo = get_edge_index(data,adj_noeye)
    return data, label, coo


def load_adj(adj_noeye):
    adj_label = sparse_mx_to_torch_sparse_tensor(adj_noeye)
    adj = adj_noeye + sp.eye(adj_noeye.shape[0])
    adj = normalize_(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, adj_label


def normalize_(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sp.coo_matrix(sparse_mx).astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def cluster_eva(epoch, y_true, y_pred):
    """
    各种评价指标，准确率，f1等
    :param y_true:
    :param y_pred:
    :return: pred_labels: 预测标签， acc: 准确率, f1: F1分数, ari: 调整兰德指数, nmi: 归一互信息化
    """
    y_pred_copy = y_pred.copy()
    y_true = y_true - np.min(y_true)  # 将 y_true 中所有标签值减去最小值，确保它们从 0 开始连续编号。

    l1 = list(set(y_true))  # 获取 y_true 中出现过的不同标签值，并将其转换为列表形式。
    numclass1 = len(l1)  # 计算 y_true 中不同标签值的数量，即真实类别数。

    l2 = list(set(y_pred_copy))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:  # 如果真实类别数和预测类别数不相等
        for i in l1:  # 遍历 l1 中的每个标签值 i
            if i in l2:  # 如果标签值 i 在 l2 中出现过，则不做任何操作
                pass
            else:  # 否则，在 y_pred 的第 ind 个位置添加标签值 i，并将 ind 加 1。
                y_pred_copy[ind] = i
                ind += 1

    l2 = list(set(y_pred_copy))  # 重新获取更新后的 y_pred 中出现过的不同标签值。
    numclass2 = len(l2)

    if numclass1 != numclass2:  # 再次检查真实类别数和预测类别数是否相等。如果不相等，则输出错误信息并返回。
        print('error')
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)  # 创建一个 numclass1 行 numclass2 列的二维数组 cost，并将其初始化为零
    # 接下来，函数创建一个二维数组 cost，其中每行对应于 y_true 中的一个标签值，每列对应于 y_pred 中的一个标签值。
    # cost[i][j] 表示 y_true 中属于第 i 类的样本中有多少个被预测为属于第 j 类。
    for i, c1 in enumerate(l1):  # 遍历 l1 中的每个标签值 c1，并获取属于第 c1 类的真实样本集合 mps
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]  # 通过列表推导式，得到所有真实标签为 c1 的样本在 y_true 中的索引列表 mps
        for j, c2 in enumerate(l2):  # 遍历 l2 中的每个标签值 c2。#
            mps_d = [i1 for i1 in mps if y_pred_copy[i1] == c2]  # 通过列表推导式，得到在 mps 中与 y_pred 中标签为 c2 的样本对应的索引列表 mps_d
            cost[i][j] = len(mps_d)  # 将 mps_d 中元素的数量赋给 cost[i][j]，即真实标签为 c1 且预测标签为 c2 的样本数量

    # match two clustering results by Munkres algorithm
    m = Munkres()  # 创建一个 Munkres 对象 m
    cost = cost.__neg__().tolist()  # 将 cost 中所有元素取相反数，以便使用 Munkres 算法最小化矩阵总和。然后，将其转换为列表形式
    indexes = m.compute(cost)  # 使用 Munkres 算法计算 cost 的最优匹配，并返回一个包含匹配结果的二元组列表 indexes

    # get the match results
    new_predict = np.zeros(len(y_pred_copy))  # 创建一个长度为 y_pred 的全零数组 new_predict，用于存储新的预测标签
    for i, c in enumerate(l1):  # 遍历 l1 中的每个标签值 c
        # correponding label in l2:
        c2 = l2[indexes[i][1]]  # 获取在 l2 中与 c 对应的标签值 c2

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred_copy) if elm == c2]  # 通过列表推导式，得到所有预测标签为 c2 的样本在 y_pred 中的索引列表 ai
        new_predict[ai] = c  # 将 new_predict 中索引为 ai 的元素赋值为 c，即将预测标签为 c2 的样本都重新赋值为真实标签 c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    precision_macro = metrics.precision_score(y_true, new_predict, average='macro')
    recall_macro = metrics.recall_score(y_true, new_predict, average='macro')
    f1_micro = metrics.f1_score(y_true, new_predict, average='micro')
    precision_micro = metrics.precision_score(y_true, new_predict, average='micro')
    recall_micro = metrics.recall_score(y_true, new_predict, average='micro')
    # Adjusted Rand index (ARI)调整兰德指数，值域是[-1,1]，负数代表结果不好，越接近于1越好
    ari = metrics.adjusted_rand_score(y_true, new_predict)
    # Normalized Mutual Information (NMI)归一化互信息，值域是[0,1]，值越高表示两个聚类结果越相似
    nmi = metrics.normalized_mutual_info_score(y_true, new_predict)
    print(epoch, ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
          ', f1 {:.4f}'.format(f1_macro))
    return acc, f1_macro, ari, nmi


def align_label(y_true, y_pred):
    y_pred_copy = y_pred.copy()
    y_true = y_true - np.min(y_true)  # 将 y_true 中所有标签值减去最小值，确保它们从 0 开始连续编号。

    l1 = list(set(y_true))  # 获取 y_true 中出现过的不同标签值，并将其转换为列表形式。
    numclass1 = len(l1)  # 计算 y_true 中不同标签值的数量，即真实类别数。

    l2 = list(set(y_pred_copy))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:  # 如果真实类别数和预测类别数不相等
        for i in l1:  # 遍历 l1 中的每个标签值 i
            if i in l2:  # 如果标签值 i 在 l2 中出现过，则不做任何操作
                pass
            else:  # 否则，在 y_pred 的第 ind 个位置添加标签值 i，并将 ind 加 1。
                y_pred_copy[ind] = i
                ind += 1

    l2 = list(set(y_pred_copy))  # 重新获取更新后的 y_pred 中出现过的不同标签值。
    numclass2 = len(l2)

    if numclass1 != numclass2:  # 再次检查真实类别数和预测类别数是否相等。如果不相等，则输出错误信息并返回。
        print('error')
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)  # 创建一个 numclass1 行 numclass2 列的二维数组 cost，并将其初始化为零
    # 接下来，函数创建一个二维数组 cost，其中每行对应于 y_true 中的一个标签值，每列对应于 y_pred 中的一个标签值。
    # cost[i][j] 表示 y_true 中属于第 i 类的样本中有多少个被预测为属于第 j 类。
    for i, c1 in enumerate(l1):  # 遍历 l1 中的每个标签值 c1，并获取属于第 c1 类的真实样本集合 mps
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]  # 通过列表推导式，得到所有真实标签为 c1 的样本在 y_true 中的索引列表 mps
        for j, c2 in enumerate(l2):  # 遍历 l2 中的每个标签值 c2。#
            mps_d = [i1 for i1 in mps if y_pred_copy[i1] == c2]  # 通过列表推导式，得到在 mps 中与 y_pred 中标签为 c2 的样本对应的索引列表 mps_d
            cost[i][j] = len(mps_d)  # 将 mps_d 中元素的数量赋给 cost[i][j]，即真实标签为 c1 且预测标签为 c2 的样本数量

    # match two clustering results by Munkres algorithm
    m = Munkres()  # 创建一个 Munkres 对象 m
    cost = cost.__neg__().tolist()  # 将 cost 中所有元素取相反数，以便使用 Munkres 算法最小化矩阵总和。然后，将其转换为列表形式
    indexes = m.compute(cost)  # 使用 Munkres 算法计算 cost 的最优匹配，并返回一个包含匹配结果的二元组列表 indexes

    # get the match results
    new_predict = np.zeros(len(y_pred_copy))  # 创建一个长度为 y_pred 的全零数组 new_predict，用于存储新的预测标签
    for i, c in enumerate(l1):  # 遍历 l1 中的每个标签值 c
        # correponding label in l2:
        c2 = l2[indexes[i][1]]  # 获取在 l2 中与 c 对应的标签值 c2

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred_copy) if elm == c2]  # 通过列表推导式，得到所有预测标签为 c2 的样本在 y_pred 中的索引列表 ai
        new_predict[ai] = c  # 将 new_predict 中索引为 ai 的元素赋值为 c，即将预测标签为 c2 的样本都重新赋值为真实标签 c
    return new_predict

class dataset(Dataset):
    def __init__(self, args):
        """
        这里的c_data和c_label均为对于的编号
        sym_in_syn包含了每个病所具有的全部症状，格式为症状的编号
        :param args:
        """
        self.one_hot = np.load(args.one_hot)
        self.data_path = args.data_path
        self.c_data_path = args.c_data_path
        self.word2ix_path = args.word2ix_path
        self.syndrome2sym_path = args.syndrome2sym_path
        with open(args.word2ix_path, "r", encoding="utf-8") as f:
            self.word2ix = json.load(f)
        self.node_embedding = self.get_node_embedding()
        self.c_data, self.c_label = self.get_c_data_label()
        self.sym_in_syn = self.get_sym_in_syn()

    def __getitem__(self, idx: int):
        return self.c_data[idx], self.c_label[idx]

    def __len__(self):
        return len(self.c_data)

    def get_sym_in_syn(self):
        with open(self.syndrome2sym_path, "r", encoding="utf-8") as f:
            syndrome2sym = json.load(f)
        syn2sym = []
        for key in syndrome2sym.keys():
            temp = []
            for item in syndrome2sym[key]:
                temp.append(self.word2ix[item])
            syn2sym.append(temp)
        return syn2sym

    def get_node_embedding(self):
        return np.load(self.data_path)
        # with open(self.data_path, 'r', encoding='utf-8') as f:
        #     embedding = json.load(f)
        # node_embedding = [embedding[key] for key in embedding.keys()]
        #
        # return node_embedding

    def get_c_data_label(self):
        data = []
        label = []
        with open(self.c_data_path, 'r', encoding='utf-8') as f:
            shuffle_data = json.load(f)

        for i in range(len(shuffle_data)):
            all_sym = []
            all_label = []
            for key in shuffle_data[i]['classify_sym'].keys():
                if key in ['厥阴病']:
                    temp_sym = shuffle_data[i]['classify_sym'][key]
                    temp_label = (np.ones(len(temp_sym)) * 4).astype(int)
                    all_sym.append(temp_sym)
                    all_label.append(temp_label)
                if key in ['少阴水饮', '少阴病']:
                    temp_sym = shuffle_data[i]['classify_sym'][key]
                    temp_label = (np.ones(len(temp_sym)) * 3).astype(int)
                    all_sym.append(temp_sym)
                    all_label.append(temp_label)
                if key in ['少阳病']:
                    temp_sym = shuffle_data[i]['classify_sym'][key]
                    temp_label = (np.ones(len(temp_sym)) * 1).astype(int)
                    all_sym.append(temp_sym)
                    all_label.append(temp_label)
                if key in ['阳明瘀血', '阳明病']:
                    temp_sym = shuffle_data[i]['classify_sym'][key]
                    temp_label = (np.ones(len(temp_sym)) * 2).astype(int)
                    all_sym.append(temp_sym)
                    all_label.append(temp_label)
                if key in ['太阴病', '太阴水饮', '太阴瘀血', '太阴痰饮']:
                    temp_sym = shuffle_data[i]['classify_sym'][key]
                    temp_label = (np.ones(len(temp_sym)) * 5).astype(int)
                    all_sym.append(temp_sym)
                    all_label.append(temp_label)
                if key in ['太阳病']:
                    temp_sym = shuffle_data[i]['classify_sym'][key]
                    temp_label = (np.ones(len(temp_sym)) * 0).astype(int)
                    all_sym.append(temp_sym)
                    all_label.append(temp_label)
            if len(all_sym) > 1:
                flatten_sym = [item for sublist in all_sym for item in sublist]
                new_flatten_sym = list(set(flatten_sym))
                # extra_sym = []
                for sym in new_flatten_sym:
                    if flatten_sym.count(sym) > 1:
                        for j in range(len(all_sym)):
                            if sym in all_sym[j]:
                                all_sym[j].remove(sym)
                                all_label[j] = np.delete(all_label[j], 0)
                flatten_all_sym = [self.word2ix[item] for sublist in all_sym for item in sublist]
                flatten_all_label = [item for sublist in all_label for item in sublist]
                data.append(flatten_all_sym)
                label.append(flatten_all_label)
            # else:
            #     flat_all_sym = [self.word2ix[item] for sublist in all_sym for item in sublist]
            #     data.append(flat_all_sym)
            #     flat_all_label = [item for sublist in all_label for item in sublist]
            #     label.append(flat_all_label)

        return data, label




class node_classify_dataset(Dataset):
    def __init__(self, data_path, word2ix, dis2idx, adj_noeye):
        self.data_path = data_path
        self.word2ix = word2ix
        self.dis2idx = dis2idx
        self.adj_noeye = adj_noeye
        self.data, self.label, self.coo = self.get_data_label()

    def __getitem__(self, idx: int):
        return self.data[idx], self.label[idx], self.coo[idx]

    def __len__(self):
        return len(self.data)

    def get_data_label(self):
        data = []
        label = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            shuffle_data = json.load(f)

        for i in range(len(shuffle_data)):
            words_to_idx = []
            for sym in shuffle_data[i]['symptoms']:
                words_to_idx.append(self.word2ix[sym])

            label_to_idx = []
            for lab in shuffle_data[i]['labels']:
                label_to_idx.append(self.dis2idx[lab])

            shuffle_idx = list(zip(words_to_idx, label_to_idx))
            random.shuffle(shuffle_idx)
            shuffled_symptoms, shuffled_labels = zip(*shuffle_idx)
            data.append(shuffled_symptoms)
            label.append(shuffled_labels)
        # adj, M_matrix = self.get_adj(data)
        coo = self.get_edge_index(data)
        return data, label, coo

    def get_edge_index(self, data):
        """
        使用官网提供的图神经网络时使用
        :param data:
        :return:
        """
        coo = []
        for d in data:
            temp_coo = []
            # edge_weights = []
            for i in range(len(d)-1):
                for j in range(i+1, len(d)):
                    adj_value = self.adj_noeye[d[i]][d[j]]
                    if adj_value != 0:
                        temp_coo.append([i, j])
                        # temp_coo.append([d[j], d[i]])
            coo_T = torch.tensor(np.transpose(np.array(temp_coo))).to(torch.int64)
            coo.append(coo_T)
            # edge_weights_array = torch.Tensor(np.array(edge_weights)).cuda()
        return coo

    def get_adj(self, data):
        """
        为n*n的邻接矩阵版GAT创建的提取adj方法
        :param data:
        :return:
        """
        adj = []
        M_matrix = []
        # data, _ = self.get_data_label()
        for d in data:
            temp_adj = [[0] * len(d) for _ in range(len(d))]
            for i in range(len(d)-1):
                for j in range(i+1, len(d)):
                    adj_value = self.adj_noeye[d[i]][d[j]]
                    if adj_value != 0:
                        temp_adj[i][j] = adj_value
                        temp_adj[j][i] = adj_value

            adj_, adj_label = load_adj(np.array(temp_adj))
            adj_dense = adj_.to_dense()
            adj_numpy = adj_dense.data.cpu().numpy()
            t = 2
            tran_prob = normalize(adj_numpy, norm="l1", axis=0)
            M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
            M = torch.Tensor(M_numpy)
            M_matrix.append(M)
            adj.append(adj_dense)
        return adj, M_matrix
