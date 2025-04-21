from basicModules import *
from torch_geometric.nn import GCNConv
from datapro import *
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

# ====================== Main Model ======================
class MHMDA(nn.Module):
    def __init__(self, param, m_emd, d_emd):
        super(MHMDA, self).__init__()
        self.inSize = param.inSize
        self.hiddenSize = param.hiddenSize
        self.outSize = param.outSize
        self.Dropout = param.Dropout
        self.device = param.device
        self.graph, self.all_meta_paths = load_dataset()

        # Multi-head hierarchical attention on meta-paths
        self.MHM = HAN_MDA(self.all_meta_paths, self.inSize, self.hiddenSize, self.outSize, self.Dropout).to(self.device)

        # Heterohypernet module
        self.HHN = HHN(param)

        self.args = param
        self.Xm = m_emd  # miRNA embedding
        self.Xd = d_emd  # Disease embedding

        # Prediction layer
        self.fcLinear = MLP(self.outSize, 1, dropout=0.1, actFunc=nn.ReLU()).to(self.device)
        self.sigmoid = nn.Sigmoid()

        # Multi-head attention for feature fusion
        self.layeratt_m = MultiHeadAttention(param.inSize, param.outSize, 2, param.num_heads1)
        self.layeratt_d = MultiHeadAttention(param.inSize, param.outSize, 2, param.num_heads1)

    def forward(self, sim_data, train_data):
        torch.manual_seed(1)

        # Initialize random features for miRNA and disease nodes
        xm = torch.randn(853, self.args.fm).to(self.device)
        xd = torch.randn(591, self.args.fd).to(self.device)

        # Multi-view similarity learning (miRNA and disease embeddings)
        Em = self.Xm(sim_data, xm)
        Ed = self.Xd(sim_data, xd)

        # "Similarity-Association-Similarity" meta-path network with hierarchical attention
        h_11, h_21 = self.MHM(self.graph, Em, Ed)

        # Extract batch node features from embeddings
        mFea, dFea = pro_data(train_data, h_11, h_21)

        # Multi-head attention fusion
        out_m = self.layeratt_m(mFea)
        out_d = self.layeratt_d(dFea)

        # Feature interaction (element-wise product)
        node_embed = out_m * out_d

        # Prediction from meta-path branch
        pre_part = self.fcLinear(node_embed)
        pre_MHM = self.sigmoid(pre_part).squeeze(dim=1)

        # Prediction from Heterohypernet
        pre_HHN = self.HHN(sim_data, train_data, Em, Ed)

        # Weighted fusion of predictions
        pre_asso = pre_MHM * 0.1 + pre_HHN * 0.9
        return pre_asso

# ====================== Data processing ======================
def pro_data(data, em, ed):
    edgeData = data.t()
    mFeaData = em
    dFeaData = ed

    m_index = edgeData[0]
    d_index = edgeData[1]

    Em = torch.index_select(mFeaData, 0, m_index)
    Ed = torch.index_select(dFeaData, 0, d_index)
    return Em, Ed

# ====================== Heterohypernet ======================
class HHN(nn.Module):
    def __init__(self, param):
        super(HHN, self).__init__()
        self.inSize = param.inSize
        self.outSize = param.outSize
        self.hiddenSize = param.hiddenSize
        self.device = param.device
        self.hdnDropout = param.hdnDropout
        self.fcDropout = param.fcDropout
        self.relu1 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # MD_hyper module
        self.md_hyper = MD_hyper(param)

        self.num_heads1 = param.num_heads1
        self.num_relations = 2  # miRNA-disease and disease-miRNA

        self.fcLinear = MLP(self.outSize, 1, dropout=self.fcDropout, actFunc=self.relu1).to(self.device)

        # Graph convolutional layer for heterogeneous message passing
        self.gcn = GCNConv(128, 128)

    def forward(self, sdata, tdata, em, ed):
        x = torch.cat([em, ed], dim=0)

        # GCN message passing on heterogeneous network
        out = torch.relu(self.gcn(x.cuda(), sdata['m_d']['edges'].cuda(),
                                  sdata['m_d']['data_matrix'][sdata['m_d']['edges'][0], sdata['m_d']['edges'][1]].cuda()))
        out1 = torch.relu(self.gcn(out.cuda(), sdata['m_d']['edges'].cuda(),
                                   sdata['m_d']['data_matrix'][sdata['m_d']['edges'][0], sdata['m_d']['edges'][1]].cuda()))

        out_m = out1[:853, :]
        out_d = out1[853:, :]

        # Extract batch node features and hyper interaction prediction
        mFea, dFea = pro_data(tdata, out_m, out_d)
        pre = self.md_hyper(mFea, dFea)
        return pre

# ====================== Hyper interaction module ======================
class MD_hyper(nn.Module):
    def __init__(self, param):
        super(MD_hyper, self).__init__()
        self.inSize = param.inSize
        self.outSize = param.outSize
        self.hiddenSize = param.hiddenSize
        self.gcnlayers = param.gcn_layers
        self.device = param.device
        self.PVN = param.PVN
        self.hdnDropout = param.hdnDropout
        self.fcDropout = param.fcDropout
        self.maskMDA = param.maskMDA
        self.realnode = param.batchSize

        self.sigmoid = nn.Sigmoid()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.LeakyReLU()
        self.num_heads1 = param.num_heads1

        # Learnable embeddings for virtual nodes
        self.nodeEmbedding = BnodeEmbedding(
            torch.tensor(np.random.normal(size=(max(int(self.PVN * self.realnode), 0), self.inSize)), dtype=torch.float32),
            dropout=self.hdnDropout).to(self.device)

        # Hypergraph convolutional network with residual connections
        self.nodeGCN = GCN(self.inSize, self.outSize, dropout=self.hdnDropout, layers=self.gcnlayers,
                           resnet=True, actFunc=self.relu1).to(self.device)

        self.fcLinear = MLP(self.outSize, 1, dropout=self.fcDropout, actFunc=self.relu1).to(self.device)
        self.layeratt_m = MultiHeadAttention(self.inSize, self.outSize, self.gcnlayers, self.num_heads1)
        self.layeratt_d = MultiHeadAttention(self.inSize, self.outSize, self.gcnlayers, self.num_heads1)

    def forward(self, em, ed):
        xm = em.unsqueeze(1)
        xd = ed.unsqueeze(1)

        if self.PVN > 0:
            # Repeat virtual node embeddings for batch
            node = self.nodeEmbedding.dropout2(self.nodeEmbedding.dropout1(self.nodeEmbedding.embedding.weight)).repeat(
                len(xd), 1, 1)
            # Concatenate real node features with virtual nodes
            node = torch.cat([xm, xd, node], dim=1)

            # Compute cosine similarity matrix among nodes
            nodeDist = torch.sqrt(torch.sum(node ** 2, dim=2, keepdim=True) + 1e-8)
            cosNode = torch.matmul(node, node.transpose(1, 2)) / (nodeDist * nodeDist.transpose(1, 2) + 1e-8)
            cosNode = self.relu2(cosNode)
            cosNode[:, range(node.shape[1]), range(node.shape[1])] = 1

            if self.maskMDA:
                cosNode[:, 0, 1] = cosNode[:, 1, 0] = 0  # optionally mask direct miRNA-disease edges

            # Normalize cosine similarity to create Laplacian matrix for hypergraph
            D = torch.eye(node.shape[1], dtype=torch.float32, device=self.device).repeat(len(xm), 1, 1)
            D[:, range(node.shape[1]), range(node.shape[1])] = 1 / (torch.sum(cosNode, dim=2) ** 0.5)
            pL = torch.matmul(torch.matmul(D, cosNode), D)

            # Hypergraph convolution
            mGCNem, dGCNem = self.nodeGCN(node, pL)

            # Multi-head attention fusion
            mLAem = self.layeratt_m(mGCNem)
            dLAem = self.layeratt_d(dGCNem)

            # Feature interaction via elementwise product
            node_embed = (mLAem * dLAem).squeeze(dim=1)
        else:
            node_embed = (xm * xd).squeeze(dim=1)

        # Classification layer
        pre_part = self.fcLinear(node_embed)
        pre_a = self.sigmoid(pre_part).squeeze(dim=1)
        return pre_a

# ====================== miRNA Embedding via Multi-view GCN ======================
class EmbeddingM(nn.Module):
    def __init__(self, args):
        super(EmbeddingM, self).__init__()
        self.args = args

        # Multi-view GCN layers for functional, sequence, and Gaussian similarity networks
        self.gcn_x1_f = GCNConv(self.args.fm, self.args.fm)
        self.gcn_x2_f = GCNConv(self.args.fm, self.args.fm)
        self.gcn_x1_s = GCNConv(self.args.fm, self.args.fm)
        self.gcn_x2_s = GCNConv(self.args.fm, self.args.fm)
        self.gcn_x1_g = GCNConv(self.args.fm, self.args.fm)
        self.gcn_x2_g = GCNConv(self.args.fm, self.args.fm)

        # Additional RNA-medicine relation paths
        self.gcn_md1 = GCNConv(self.args.fm, self.args.fm)
        self.gcn_md2 = GCNConv(self.args.fm, self.args.fm)

        # Channel-wise attention layers
        self.fc1_x = nn.Linear(self.args.view * self.args.gcn_layers1,
                               5 * self.args.view * self.args.gcn_layers1)
        self.fc2_x = nn.Linear(5 * self.args.view * self.args.gcn_layers1,
                               self.args.view * self.args.gcn_layers1)

        self.sigmoidx = nn.Sigmoid()

        # 1x1 convolution to compress feature maps after attention
        self.cnn_x = nn.Conv2d(self.args.view * self.args.gcn_layers1, 1, kernel_size=(1, 1), stride=1, bias=True)

    def forward(self, data, fm1):
        # Multi-view GCN feature extraction per similarity type
        x_f1 = torch.relu(self.gcn_x1_f(fm1.cuda(), data['mm_f']['edges'].cuda(),
                          data['mm_f']['data_matrix'][data['mm_f']['edges'][0], data['mm_f']['edges'][1]].cuda()))
        x_f2 = torch.relu(self.gcn_x2_f(x_f1, data['mm_f']['edges'].cuda(),
                          data['mm_f']['data_matrix'][data['mm_f']['edges'][0], data['mm_f']['edges'][1]].cuda()))

        x_s1 = torch.relu(self.gcn_x1_s(fm1.cuda(), data['mm_s']['edges'].cuda(),
                          data['mm_s']['data_matrix'][data['mm_s']['edges'][0], data['mm_s']['edges'][1]].cuda()))
        x_s2 = torch.relu(self.gcn_x2_s(x_s1, data['mm_s']['edges'].cuda(),
                          data['mm_s']['data_matrix'][data['mm_s']['edges'][0], data['mm_s']['edges'][1]].cuda()))

        x_g1 = torch.relu(self.gcn_x1_g(fm1.cuda(), data['mm_g']['edges'].cuda(),
                          data['mm_g']['data_matrix'][data['mm_g']['edges'][0], data['mm_g']['edges'][1]].cuda()))
        x_g2 = torch.relu(self.gcn_x2_g(x_g1, data['mm_g']['edges'].cuda(),
                          data['mm_g']['data_matrix'][data['mm_g']['edges'][0], data['mm_g']['edges'][1]].cuda()))

        # Concatenate multi-view features
        XM = torch.cat((x_f1, x_f2, x_s1, x_s2, x_g1, x_g2), dim=1).t()

        # Reshape for channel attention
        XM = XM.view(1, self.args.view * self.args.gcn_layers1, self.args.fm, -1)

        # Channel attention mechanism
        globalAvgPool_x = nn.AvgPool2d((self.args.fm, 853), (1, 1))
        x_channel_attention = globalAvgPool_x(XM).view(1, -1)

        x_channel_attention = torch.relu(self.fc1_x(x_channel_attention))
        x_channel_attention = self.sigmoidx(self.fc2_x(x_channel_attention))

        x_channel_attention = x_channel_attention.view(1, -1, 1, 1)

        # Reweight features by attention
        XM_channel_attention = torch.relu(x_channel_attention * XM)

        # Feature compression to output embedding dimension
        y = self.cnn_x(XM_channel_attention).view(self.args.fm, 853).t()

        return y

# ====================== Disease Embedding via Multi-view GCN ======================
class EmbeddingD(nn.Module):
    def __init__(self, args):
        super(EmbeddingD, self).__init__()
        self.args = args

        # Multi-view GCN layers for temporal, spatial and gaussian similarities
        self.gcn_y1_t = GCNConv(self.args.fd, self.args.fd)
        self.gcn_y2_t = GCNConv(self.args.fd, self.args.fd)
        self.gcn_y1_s = GCNConv(self.args.fd, self.args.fd)
        self.gcn_y2_s = GCNConv(self.args.fd, self.args.fd)
        self.gcn_y1_g = GCNConv(self.args.fd, self.args.fd)
        self.gcn_y2_g = GCNConv(self.args.fd, self.args.fd)

        # Additional RNA-disease relation paths
        self.gcn_rd1 = GCNConv(self.args.fd, self.args.fd)
        self.gcn_rd2 = GCNConv(self.args.fd, self.args.fd)

        # Channel-wise attention layers
        self.fc1_y = nn.Linear(self.args.view * self.args.gcn_layers1,
                               5 * self.args.view * self.args.gcn_layers1)
        self.fc2_y = nn.Linear(5 * self.args.view * self.args.gcn_layers1,
                               self.args.view * self.args.gcn_layers1)

        self.sigmoidy = nn.Sigmoid()

        # 1x1 convolution for feature compression
        self.cnn_y = nn.Conv2d(self.args.view * self.args.gcn_layers1, 1, kernel_size=(1, 1), stride=1, bias=True)

    def forward(self, data, dm1):
        # Multi-view GCN feature extraction
        y_d_t1 = torch.relu(self.gcn_y1_t(dm1.cuda(), data['dd_t']['edges'].cuda(),
                                          data['dd_t']['data_matrix'][data['dd_t']['edges'][0], data['dd_t']['edges'][1]].cuda()))
        y_d_t2 = torch.relu(self.gcn_y2_t(y_d_t1, data['dd_t']['edges'].cuda(),
                                          data['dd_t']['data_matrix'][data['dd_t']['edges'][0], data['dd_t']['edges'][1]].cuda()))
        y_d_s1 = torch.relu(self.gcn_y1_s(dm1.cuda(), data['dd_s']['edges'].cuda(),
                                          data['dd_s']['data_matrix'][data['dd_s']['edges'][0], data['dd_s']['edges'][1]].cuda()))
        y_d_s2 = torch.relu(self.gcn_y2_s(y_d_s1, data['dd_s']['edges'].cuda(),
                                          data['dd_s']['data_matrix'][data['dd_s']['edges'][0], data['dd_s']['edges'][1]].cuda()))
        y_d_g1 = torch.relu(self.gcn_y1_g(dm1.cuda(), data['dd_g']['edges'].cuda(),
                                          data['dd_g']['data_matrix'][data['dd_g']['edges'][0], data['dd_g']['edges'][1]].cuda()))
        y_d_g2 = torch.relu(self.gcn_y2_g(y_d_g1, data['dd_g']['edges'].cuda(),
                                          data['dd_g']['data_matrix'][data['dd_g']['edges'][0], data['dd_g']['edges'][1]].cuda()))

        # Concatenate multi-view features
        combined_embedding = torch.cat((y_d_t1, y_d_t2, y_d_s1, y_d_s2, y_d_g1, y_d_g2), 1).t()

        combined_embedding = combined_embedding.view(1, self.args.view * self.args.gcn_layers1, self.args.fd, -1)

        # Channel attention mechanism
        globalAvgPool_y = nn.AvgPool2d((self.args.fd, 591), (1, 1))
        y_channel_attention = globalAvgPool_y(combined_embedding).view(1, -1)

        y_channel_attention = torch.relu(self.fc1_y(y_channel_attention))
        y_channel_attention = self.sigmoidy(self.fc2_y(y_channel_attention))

        y_channel_attention = y_channel_attention.view(1, -1, 1, 1)

        # Reweight features by channel attention
        YD_channel_attention = torch.relu(y_channel_attention * combined_embedding)

        # Compress features to final output embedding dimension
        y = self.cnn_y(YD_channel_attention).view(self.args.fd, 591).t()

        return y

# ====================== Semantic-level Attention ======================
class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=32):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size).apply(init),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False).apply(init)
        )

    def forward(self, z):
        # z shape: (batch, num_metapaths, hidden_size)
        w = self.project(z).mean(0)    # Path-wise attention logits
        beta = torch.softmax(w, dim=0) # Attention weights over meta-paths

        beta = beta.expand((z.shape[0],) + beta.shape)  # Broadcast

        return (beta * z).sum(1)  # Weighted sum semantic embedding

def dgl_to_pyg(g, h):
    src, dst = g.edges()
    edge_index = torch.stack([src, dst], dim=0).to(g.device)
    x = h
    return Data(x=x, edge_index=edge_index)

# ====================== HANLayer ======================
class HANLayer(nn.Module):
    def __init__(self, meta_paths, in_size, out_size, dropout):
        super(HANLayer, self).__init__()
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
        self.gat_layers = nn.ModuleList([
            GATConv(
                in_channels=in_size,
                out_channels=out_size,
                heads=2,
                dropout=dropout,
                concat=True,
                negative_slope=0.2,
                add_self_loops=True,
            ) for _ in range(len(meta_paths))
        ])

        self._cached_graph = None
        self._cached_coalesced_graph = {}
        self.semantic_attention = SemanticAttention(in_size=out_size * 2)

    def forward(self, g, h):
        # Cache and generate metapath-based subgraphs
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(g, meta_path)

        device = h.device
        semantic_embeddings = []

        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path].to(device)
            pyg_data = dgl_to_pyg(new_g, h)
            out = self.gat_layers[i](pyg_data.x, pyg_data.edge_index)
            semantic_embeddings.append(out.flatten(1))

        semantic_embeddings_stack = torch.stack(semantic_embeddings, dim=1)

        return self.semantic_attention(semantic_embeddings_stack)  # Shape: (N, out_size*2)

# ====================== HAN module ======================
class HAN(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, out_size, dropout):
        super(HAN, self).__init__()
        self.layer = HANLayer(meta_paths, in_size, hidden_size, dropout)
        self.predict = nn.Linear(64, out_size, bias=False).apply(init)

    def forward(self, g, h):
        h = self.layer(g, h)
        return self.predict(h)

# ====================== Dual-channel HAN for miRNA and Disease ======================
class HAN_MDA(nn.Module):
    def __init__(self, all_meta_paths, in_size, hidden_size, out_size, dropout):
        super(HAN_MDA, self).__init__()
        self.sum_layers = nn.ModuleList([
            HAN(all_meta_paths[i], in_size, hidden_size, out_size, dropout) for i in range(len(all_meta_paths))
        ])

    def forward(self, s_g, s_h_1, s_h_2):
        h_11 = self.sum_layers[0](s_g[0], s_h_1)
        h_21 = self.sum_layers[1](s_g[1], s_h_2)
        return h_11, h_21

def init(i):
    if isinstance(i, nn.Linear):
        torch.nn.init.xavier_uniform_(i.weight)

# ====================== Multi-Head Attention ======================
class MultiHeadAttention(nn.Module):
    def __init__(self, inSize, outSize, gcnlayers, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([LayerAtt2(inSize, outSize) for _ in range(num_heads)])
        self.merge_layer = nn.Linear(num_heads * outSize, outSize)

    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]
        concatenated = torch.cat(head_outputs, dim=-1)  # Concatenate heads on last dim
        return self.merge_layer(concatenated)
