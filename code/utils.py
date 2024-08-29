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
    :param y_true:
    :param y_pred:
    :return: pred_labels, acc, f1, ari, nmi
    """
    y_pred_copy = y_pred.copy()
    y_true = y_true - np.min(y_true)  

    l1 = list(set(y_true)) 
    numclass1 = len(l1)

    l2 = list(set(y_pred_copy))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2: 
        for i in l1:  
            if i in l2:  
                pass
            else:  
                y_pred_copy[ind] = i
                ind += 1

    l2 = list(set(y_pred_copy)) 
    numclass2 = len(l2)

    if numclass1 != numclass2: 
        print('error')
        return

    cost = np.zeros((numclass1, numclass2), dtype=int) 
    for i, c1 in enumerate(l1):  
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]  
        for j, c2 in enumerate(l2):  
            mps_d = [i1 for i1 in mps if y_pred_copy[i1] == c2]  
            cost[i][j] = len(mps_d)  

    # match two clustering results by Munkres algorithm
    m = Munkres() 
    cost = cost.__neg__().tolist() 
    indexes = m.compute(cost)  

    # get the match results
    new_predict = np.zeros(len(y_pred_copy))  
    for i, c in enumerate(l1):  
        # correponding label in l2:
        c2 = l2[indexes[i][1]] 

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred_copy) if elm == c2]  
        new_predict[ai] = c 

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    precision_macro = metrics.precision_score(y_true, new_predict, average='macro')
    recall_macro = metrics.recall_score(y_true, new_predict, average='macro')
    f1_micro = metrics.f1_score(y_true, new_predict, average='micro')
    precision_micro = metrics.precision_score(y_true, new_predict, average='micro')
    recall_micro = metrics.recall_score(y_true, new_predict, average='micro')
    # Adjusted Rand index (ARI)
    ari = metrics.adjusted_rand_score(y_true, new_predict)
    # Normalized Mutual Information (NMI)
    nmi = metrics.normalized_mutual_info_score(y_true, new_predict)
    print(epoch, ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
          ', f1 {:.4f}'.format(f1_macro))
    return acc, f1_macro, ari, nmi


def align_label(y_true, y_pred):
    y_pred_copy = y_pred.copy()
    y_true = y_true - np.min(y_true) 

    l1 = list(set(y_true))
    numclass1 = len(l1) 

    l2 = list(set(y_pred_copy))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2: 
        for i in l1:  
            if i in l2: 
                pass
            else:  
                y_pred_copy[ind] = i
                ind += 1

    l2 = list(set(y_pred_copy)) 
    numclass2 = len(l2)

    if numclass1 != numclass2: 
        print('error')
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)  

    for i, c1 in enumerate(l1):  
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1] 
        for j, c2 in enumerate(l2):  
            mps_d = [i1 for i1 in mps if y_pred_copy[i1] == c2]  
            cost[i][j] = len(mps_d)  

    # match two clustering results by Munkres algorithm
    m = Munkres()  
    cost = cost.__neg__().tolist()  
    indexes = m.compute(cost)  

    # get the match results
    new_predict = np.zeros(len(y_pred_copy))  
    for i, c in enumerate(l1):  
        # correponding label in l2:
        c2 = l2[indexes[i][1]]  

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred_copy) if elm == c2]  
        new_predict[ai] = c  
    return new_predict

class dataset(Dataset):
    def __init__(self, args):
        """
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
