import dgl
import numpy as np
from scipy.sparse import coo_matrix
import os
import torch
import csv
import torch.utils.data.dataset as Dataset
from scipy import sparse
import pandas as pd


def dense2sparse(matrix: np.ndarray):
    """
    Convert dense matrix to COO sparse representation (edge indices and values).
    """
    mat_coo = coo_matrix(matrix)
    edge_idx = np.vstack((mat_coo.row, mat_coo.col))
    return edge_idx, mat_coo.data


def loading_data(param):
    """
    Load miRNA-disease association matrix and generate train/test positive and negative samples.
    Splitting is controlled by param.ratio.
    """
    ratio = param.ratio
    md_matrix = np.loadtxt(os.path.join(param.datapath + '/m_d.csv'), dtype=int, delimiter=',')

    # Positive samples (existing associations)
    rng = np.random.default_rng(seed=42)
    pos_samples = np.where(md_matrix == 1)
    pos_samples_shuffled = rng.permutation(pos_samples, axis=1)

    # Negative samples (non-associations), sample balanced to positive count
    rng = np.random.default_rng(seed=42)
    neg_samples = np.where(md_matrix == 0)
    neg_samples_shuffled = rng.permutation(neg_samples, axis=1)[:, :pos_samples_shuffled.shape[1]]

    edge_idx_dict = dict()
    n_pos_samples = pos_samples_shuffled.shape[1]

    idx_split = int(n_pos_samples * ratio)

    # Split positive and negative edges for testing and training
    test_pos_edges = pos_samples_shuffled[:, :idx_split].T
    test_neg_edges = neg_samples_shuffled[:, :idx_split].T
    test_true_label = np.array(np.hstack((np.ones(test_pos_edges.shape[0]), np.zeros(test_neg_edges.shape[0]))), dtype='float32')
    test_edges = np.vstack((test_pos_edges, test_neg_edges))

    train_pos_edges = pos_samples_shuffled[:, idx_split:].T
    train_neg_edges = neg_samples_shuffled[:, idx_split:].T
    train_true_label = np.array(np.hstack((np.ones(train_pos_edges.shape[0]), np.zeros(train_neg_edges.shape[0]))), dtype='float32')
    train_edges = np.vstack((train_pos_edges, train_neg_edges))

    edge_idx_dict['train_Edges'] = train_edges
    edge_idx_dict['train_Labels'] = train_true_label

    edge_idx_dict['test_Edges'] = test_edges
    edge_idx_dict['test_Labels'] = test_true_label

    edge_idx_dict['true_md'] = md_matrix

    return edge_idx_dict


def read_csv(path):
    """
    Load similarity or association matrix from CSV and convert to torch.Tensor.
    """
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return torch.Tensor(md_data)


def get_edge_index(matrix):
    """
    Get non-zero indices of a torch matrix as edge_index in (2, E) shape.
    """
    non_zero_coords = matrix.nonzero(as_tuple=True)
    edge_index = torch.stack(non_zero_coords, dim=0)
    return edge_index


def Simdata_pro(param):
    """
    Read various similarity matrices and construct corresponding data dictionaries
    with data matrix and edge indices for graph construction.
    Also constructs a combined miRNA-disease association matrix as a bipartite graph adjacency.
    """
    dataset = dict()

    # miRNA sequence similarity
    mm_s_matrix = read_csv(param.datapath + '/m_ss.csv')
    mm_s_edge_index = get_edge_index(mm_s_matrix)
    dataset['mm_s'] = {'data_matrix': mm_s_matrix, 'edges': mm_s_edge_index}

    # Disease target-based similarity
    dd_t_matrix = read_csv(param.datapath + '/d_ts.csv')
    dd_t_edge_index = get_edge_index(dd_t_matrix)
    dataset['dd_t'] = {'data_matrix': dd_t_matrix, 'edges': dd_t_edge_index}

    # miRNA functional similarity
    mm_f_matrix = read_csv(param.datapath + '/m_fs.csv')
    mm_f_edge_index = get_edge_index(mm_f_matrix)
    dataset['mm_f'] = {'data_matrix': mm_f_matrix, 'edges': mm_f_edge_index}

    # Disease semantic similarity
    dd_s_matrix = read_csv(param.datapath + '/d_ss.csv')
    dd_s_edge_index = get_edge_index(dd_s_matrix)
    dataset['dd_s'] = {'data_matrix': dd_s_matrix, 'edges': dd_s_edge_index}

    # miRNA Gaussian similarity
    mm_g_matrix = read_csv(param.datapath + '/m_gs.csv')
    mm_g_edge_index = get_edge_index(mm_g_matrix)
    dataset['mm_g'] = {'data_matrix': mm_g_matrix, 'edges': mm_g_edge_index}

    # Disease Gaussian similarity
    dd_g_matrix = read_csv(param.datapath + '/d_gs.csv')
    dd_g_edge_index = get_edge_index(dd_g_matrix)
    dataset['dd_g'] = {'data_matrix': dd_g_matrix, 'edges': dd_g_edge_index}

    # miRNA-disease association matrix combined as bipartite adjacency
    md_matrix = read_csv(param.datapath + '/m_d.csv')
    zero_matrix_rna = torch.zeros((853, 853))
    zero_matrix_disease = torch.zeros((591, 591))

    md_matrix_transposed = torch.t(md_matrix)

    top_half = torch.cat([zero_matrix_rna, md_matrix], dim=1)
    bottom_half = torch.cat([md_matrix_transposed, zero_matrix_disease], dim=1)

    md_matrix = torch.cat([top_half, bottom_half], dim=0)

    md_edge_index = get_edge_index(md_matrix)
    dataset['m_d'] = {'data_matrix': md_matrix, 'edges': md_edge_index}

    return dataset


def load_dataset():
    """
    Load multiple miRNA and disease similarity matrices from CSV files,
    perform matrix multiplications to derive cross-type similarity features,
    binarize similarities with threshold 0.5,
    and create heterogeneous DGL graphs representing different relations.
    """
    m_d = pd.read_csv('./datasets/m_d.csv', delimiter=',', header=None).values
    d_m = m_d.T

    m_ss = pd.read_csv('./datasets/m_ss.csv', delimiter=',', header=None).values
    m_fs = pd.read_csv('./datasets/m_fs.csv', delimiter=',', header=None).values
    m_gs = pd.read_csv('./datasets/m_gs.csv', delimiter=',', header=None).values

    d_ts = pd.read_csv('./datasets/d_ts.csv', delimiter=',', header=None).values
    d_ss = pd.read_csv('./datasets/d_ss.csv', delimiter=',', header=None).values
    d_gs = pd.read_csv('./datasets/d_gs.csv', delimiter=',', header=None).values

    # Compute cross-type similarities via matrix multiplication
    m_ss_D = m_d.dot(d_ss)
    D_ss_m = m_ss_D.T

    m_gs_D = m_d.dot(d_gs)
    D_gs_m = m_gs_D.T

    m_ts_D = m_d.dot(d_ts)
    D_ts_m = m_ts_D.T

    M_ss_d = m_ss.dot(m_d)
    d_ss_M = M_ss_d.T

    M_gs_d = m_gs.dot(m_d)
    d_gs_M = M_gs_d.T

    M_fs_d = m_fs.dot(m_d)
    d_fs_M = M_fs_d.T

    # Binarize similarity matrices using threshold 0.5
    m_ss = (m_ss >= 0.5).astype(int)
    m_fs = (m_fs >= 0.5).astype(int)
    m_gs = (m_gs >= 0.5).astype(int)

    d_ts = (d_ts >= 0.5).astype(int)
    d_ss = (d_ss >= 0.5).astype(int)
    d_gs = (d_gs >= 0.5).astype(int)

    M_ss_d = (M_ss_d >= 0.5).astype(int)
    d_ss_M = (d_ss_M >= 0.5).astype(int)
    M_fs_d = (M_fs_d >= 0.5).astype(int)
    d_fs_M = (d_fs_M >= 0.5).astype(int)
    M_gs_d = (M_gs_d >= 0.5).astype(int)
    d_gs_M = (d_gs_M >= 0.5).astype(int)

    D_ts_m = (D_ts_m >= 0.5).astype(int)
    m_ts_D = (m_ts_D >= 0.5).astype(int)
    D_ss_m = (D_ss_m >= 0.5).astype(int)
    m_ss_D = (m_ss_D >= 0.5).astype(int)
    D_gs_m = (D_gs_m >= 0.5).astype(int)
    m_gs_D = (m_gs_D >= 0.5).astype(int)

    # Construct heterogeneous graphs for miRNA similarity relations
    mg1 = {('miRNA', 'm_ss_m', 'miRNA'): sparse.csr_matrix(m_ss).nonzero()}
    mg2 = {('miRNA', 'm_gs_m', 'miRNA'): sparse.csr_matrix(m_gs).nonzero()}
    mg3 = {('miRNA', 'm_fs_m', 'miRNA'): sparse.csr_matrix(m_fs).nonzero()}

    # Construct heterogeneous graphs for disease similarity relations
    dg1 = {('disease', 'd_ss_d', 'disease'): sparse.csr_matrix(d_ss).nonzero()}
    dg2 = {('disease', 'd_gs_d', 'disease'): sparse.csr_matrix(d_gs).nonzero()}
    dg3 = {('disease', 'd_ts_d', 'disease'): sparse.csr_matrix(d_ts).nonzero()}

    # Construct miRNA to disease relations graphs
    m_d1 = {('miRNA', 'M_ss_d', 'disease'): sparse.csr_matrix(M_ss_d).nonzero()}
    m_d2 = {('miRNA', 'm_ss_D', 'disease'): sparse.csr_matrix(m_ss_D).nonzero()}
    m_d3 = {('miRNA', 'M_gs_d', 'disease'): sparse.csr_matrix(M_gs_d).nonzero()}
    m_d4 = {('miRNA', 'm_gs_D', 'disease'): sparse.csr_matrix(m_gs_D).nonzero()}
    m_d5 = {('miRNA', 'M_fs_d', 'disease'): sparse.csr_matrix(M_fs_d).nonzero()}
    m_d6 = {('miRNA', 'm_ts_D', 'disease'): sparse.csr_matrix(m_ts_D).nonzero()}

    # Construct disease to miRNA (inverse) relations
    d_m1 = {('disease', 'd_ss_M', 'miRNA'): sparse.csr_matrix(d_ss_M).nonzero()}
    d_m2 = {('disease', 'D_ss_m', 'miRNA'): sparse.csr_matrix(D_ss_m).nonzero()}
    d_m3 = {('disease', 'd_gs_M', 'miRNA'): sparse.csr_matrix(d_gs_M).nonzero()}
    d_m4 = {('disease', 'D_gs_m', 'miRNA'): sparse.csr_matrix(D_gs_m).nonzero()}
    d_m5 = {('disease', 'd_fs_M', 'miRNA'): sparse.csr_matrix(d_fs_M).nonzero()}
    d_m6 = {('disease', 'D_ts_m', 'miRNA'): sparse.csr_matrix(D_ts_m).nonzero()}

    # Build DGL heterogeneous graphs for each relation
    mg1 = dgl.heterograph(mg1)
    mg2 = dgl.heterograph(mg2)
    mg3 = dgl.heterograph(mg3)

    dg1 = dgl.heterograph(dg1)
    dg2 = dgl.heterograph(dg2)
    dg3 = dgl.heterograph(dg3)

    m_d1 = dgl.heterograph(m_d1)
    m_d2 = dgl.heterograph(m_d2)
    m_d3 = dgl.heterograph(m_d3)
    m_d4 = dgl.heterograph(m_d4)
    m_d5 = dgl.heterograph(m_d5)
    m_d6 = dgl.heterograph(m_d6)

    d_m1 = dgl.heterograph(d_m1)
    d_m2 = dgl.heterograph(d_m2)
    d_m3 = dgl.heterograph(d_m3)
    d_m4 = dgl.heterograph(d_m4)
    d_m5 = dgl.heterograph(d_m5)
    d_m6 = dgl.heterograph(d_m6)

    # Extract edge indices for constructing combined heterographs
    mg1_src, mg1_dst = mg1.edges(etype='m_ss_m')
    mg2_src, mg2_dst = mg2.edges(etype='m_gs_m')
    mg3_src, mg3_dst = mg3.edges(etype='m_fs_m')

    dg1_src, dg1_dst = dg1.edges(etype='d_ss_d')
    dg2_src, dg2_dst = dg2.edges(etype='d_gs_d')
    dg3_src, dg3_dst = dg3.edges(etype='d_ts_d')

    m_d1_src, m_d1_dst = m_d1.edges(etype='M_ss_d')
    m_d2_src, m_d2_dst = m_d2.edges(etype='m_ss_D')
    m_d3_src, m_d3_dst = m_d3.edges(etype='M_gs_d')
    m_d4_src, m_d4_dst = m_d4.edges(etype='m_gs_D')
    m_d5_src, m_d5_dst = m_d5.edges(etype='M_fs_d')
    m_d6_src, m_d6_dst = m_d6.edges(etype='m_ts_D')

    d_m1_src, d_m1_dst = d_m1.edges(etype='d_ss_M')
    d_m2_src, d_m2_dst = d_m2.edges(etype='D_ss_m')
    d_m3_src, d_m3_dst = d_m3.edges(etype='d_gs_M')
    d_m4_src, d_m4_dst = d_m4.edges(etype='D_gs_m')
    d_m5_src, d_m5_dst = d_m5.edges(etype='d_fs_M')
    d_m6_src, d_m6_dst = d_m6.edges(etype='D_ts_m')

    # Combine edges for miRNA and disease heterograph 1
    combined_mg = {
        ('miRNA', 'm_ss_m', 'miRNA'): (mg1_src, mg1_dst),
        ('miRNA', 'm_gs_m', 'miRNA'): (mg3_src, mg3_dst),
        ('miRNA', 'm_fs_m', 'miRNA'): (mg2_src, mg2_dst),
        ('miRNA', 'M_ss_d', 'disease'): (m_d1_src, m_d1_dst),
        ('disease', 'd_ss_M', 'miRNA'): (d_m1_src, d_m1_dst),
        ('miRNA', 'M_gs_d', 'disease'): (m_d3_src, m_d3_dst),
        ('disease', 'd_gs_M', 'miRNA'): (d_m3_src, d_m3_dst),
        ('miRNA', 'M_fs_d', 'disease'): (m_d5_src, m_d5_dst),
        ('disease', 'd_fs_M', 'miRNA'): (d_m5_src, d_m5_dst),
    }

    # Combine edges for heterograph 2 related mainly to disease
    combined_dg = {
        ('disease', 'd_ss_d', 'disease'): (dg2_src, dg2_dst),
        ('disease', 'd_gs_d', 'disease'): (dg3_src, dg3_dst),
        ('disease', 'd_ts_d', 'disease'): (dg1_src, dg1_dst),
        ('disease', 'D_ss_m', 'miRNA'): (d_m2_src, d_m2_dst),
        ('miRNA', 'm_ss_D', 'disease'): (m_d2_src, m_d2_dst),
        ('disease', 'D_gs_m', 'miRNA'): (d_m4_src, d_m4_dst),
        ('miRNA', 'm_gs_D', 'disease'): (m_d4_src, m_d4_dst),
        ('disease', 'D_ts_m', 'miRNA'): (d_m6_src, d_m6_dst),
        ('miRNA', 'm_ts_D', 'disease'): (m_d6_src, m_d6_dst),
    }

    mg = dgl.heterograph(combined_mg)
    dg = dgl.heterograph(combined_dg)

    graph = [mg, dg]

    # Define "Similarity-Association-Similarity" meta-paths for heterogeneous graph learning
    all_meta_paths = [
        [['M_ss_d', 'd_ss_M'], ['M_gs_d', 'd_gs_M'], ['M_fs_d', 'd_fs_M']],
        [['D_ss_m', 'm_ss_D'], ['D_gs_m', 'm_gs_D'], ['D_ts_m', 'm_ts_D']]
    ]

    return graph, all_meta_paths


class CVEdgeDataset(Dataset.Dataset):
    """
    Dataset class for cross-validation of edge prediction.
    Stores edges and corresponding labels.
    """
    def __init__(self, edges, labels):
        self.Data = edges
        self.Label = labels

    def __len__(self):
        return len(self.Label)

    def __getitem__(self, index):
        data = self.Data[index]
        label = self.Label[index]
        return data, label
