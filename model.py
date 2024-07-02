from basicModules import *
from torch_geometric.nn import GCNConv
from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F
from datapro import *


class MHMDA(nn.Module):
    def __init__(self, param, m_emd, d_emd):
        super(MHMDA, self).__init__()
        self.inSize = param.inSize
        self.hiddenSize = param.hiddenSize
        self.outSize = param.outSize
        self.Dropout = param.Dropout
        self.device = param.device
        self.graph, self.all_meta_paths = load_dataset()
        self.MHM = HAN_DTI(self.all_meta_paths, self.inSize, self.hiddenSize, self.outSize, self.Dropout).to(
            self.device)
        self.HHN = HHN(param)
        self.args = param
        self.Xm = m_emd
        self.Xd = d_emd
        self.fcLinear = MLP(self.outSize, 1, dropout=0.1, actFunc=nn.ReLU()).to(self.device)
        self.sigmoid = nn.Sigmoid()
        self.layeratt_m = MultiHeadAttention(param.inSize, param.outSize, 2, param.num_heads1)
        self.layeratt_d = MultiHeadAttention(param.inSize, param.outSize, 2, param.num_heads1)

    def forward(self, sim_data, train_data):
        torch.manual_seed(1)
        xm = torch.randn(853, self.args.fm).to(self.device)
        xd = torch.randn(591, self.args.fd).to(self.device)
        Em = self.Xm(sim_data, xm)
        Ed = self.Xd(sim_data, xd)
        h_11, h_21 = self.MHM(self.graph, Em, Ed)
        mFea, dFea = pro_data(train_data, h_11, h_21)
        out_m = self.layeratt_m(mFea)
        out_d = self.layeratt_d(dFea)
        node_embed = out_m * out_d
        pre_part = self.fcLinear(node_embed)
        pre_asso1 = self.sigmoid(pre_part).squeeze(dim=1)
        pre_asso2 = self.HHN(sim_data, train_data, Em, Ed)
        pre_asso = pre_asso1 * 0.4 + pre_asso2 * 0.6
        return pre_asso


def pro_data(data, em, ed):
    edgeData = data.t()
    mFeaData = em
    dFeaData = ed
    m_index = edgeData[0]
    d_index = edgeData[1]
    Em = torch.index_select(mFeaData, 0, m_index)
    Ed = torch.index_select(dFeaData, 0, d_index)
    return Em, Ed


class HHN(nn.Module):
    def __init__(self, param):
        super(HHN, self).__init__()
        self.inSize = param.inSize
        self.outSize = param.outSize
        self.hiddenSize = param.hiddenSize
        self.device = param.device
        self.hdnDropout = param.hdnDropout
        self.fcDropout = param.fcDropout
        self.sigmoid = nn.Sigmoid()
        self.relu1 = nn.ReLU()
        self.md_supernode = MDI(param)

        self.num_heads1 = param.num_heads1
        self.num_relations = 2  # Assuming 2 relations, one for rna to disease and one for disease to rna

        self.fcLinear = MLP(self.outSize, 1, dropout=self.fcDropout, actFunc=self.relu1).to(self.device)
        self.layeratt_m = MultiHeadAttention(self.inSize, self.outSize, self.num_relations, self.num_heads1)
        self.layeratt_d = MultiHeadAttention(self.inSize, self.outSize, self.num_relations, self.num_heads1)
        self.gcn = GCNConv(128, 128)

    def forward(self, sdata, tdata, em, ed):
        x = (torch.cat([em, ed], dim=0))
        out = torch.relu(self.gcn(x.cuda(), sdata['m_d']['edges'].cuda(), sdata['m_d']['data_matrix'][
            sdata['m_d']['edges'][0], sdata['m_d']['edges'][1]].cuda()))
        out1 = torch.relu(self.gcn(out.cuda(), sdata['m_d']['edges'].cuda(), sdata['m_d']['data_matrix'][
            sdata['m_d']['edges'][0], sdata['m_d']['edges'][1]].cuda()))
        out_m = out1[:853, :]
        out_d = out1[853:, :]
        mFea, dFea = pro_data(tdata, out_m, out_d)
        pre = self.md_supernode(mFea, dFea)
        return pre


class MDI(nn.Module):
    def __init__(self, param):
        super(MDI, self).__init__()

        self.inSize = param.inSize
        self.outSize = param.outSize
        self.hiddenSize = param.hiddenSize
        self.gcnlayers = param.gcn_layers
        self.device = param.device
        self.nodeNum = param.nodeNum
        self.hdnDropout = param.hdnDropout
        self.fcDropout = param.fcDropout
        self.maskMDI = param.maskMDI
        self.sigmoid = nn.Sigmoid()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.LeakyReLU()
        self.num_heads1 = param.num_heads1

        self.nodeEmbedding = BnodeEmbedding(
            torch.tensor(np.random.normal(size=(max(self.nodeNum, 0), self.inSize)), dtype=torch.float32),
            dropout=self.hdnDropout).to(self.device)
        self.nodeGCN = GCN(self.inSize, self.outSize, dropout=self.hdnDropout, layers=self.gcnlayers, resnet=True,
                           actFunc=self.relu1).to(self.device)

        self.fcLinear = MLP(self.outSize, 1, dropout=self.fcDropout, actFunc=self.relu1).to(self.device)
        self.layeratt_m = MultiHeadAttention(self.inSize, self.outSize, self.gcnlayers, self.num_heads1)
        self.layeratt_d = MultiHeadAttention(self.inSize, self.outSize, self.gcnlayers, self.num_heads1)

    def forward(self, em, ed):
        xm = em.unsqueeze(1)
        xd = ed.unsqueeze(1)
        if self.nodeNum > 0:
            node = self.nodeEmbedding.dropout2(self.nodeEmbedding.dropout1(self.nodeEmbedding.embedding.weight)).repeat(
                len(xd), 1, 1)
            node = torch.cat([xm, xd, node], dim=1)
            nodeDist = torch.sqrt(torch.sum(node ** 2, dim=2, keepdim=True) + 1e-8)
            cosNode = torch.matmul(node, node.transpose(1, 2)) / (nodeDist * nodeDist.transpose(1, 2) + 1e-8)
            cosNode = self.relu2(cosNode)
            cosNode[:, range(node.shape[1]), range(node.shape[1])] = 1
            if self.maskMDI: cosNode[:, 0, 1] = cosNode[:, 1, 0] = 0
            D = torch.eye(node.shape[1], dtype=torch.float32, device=self.device).repeat(len(xm), 1, 1)
            D[:, range(node.shape[1]), range(node.shape[1])] = 1 / (torch.sum(cosNode, dim=2) ** 0.5)
            pL = torch.matmul(torch.matmul(D, cosNode), D)

            mGCNem, dGCNem = self.nodeGCN(node, pL)
            mLAem = self.layeratt_m(mGCNem)
            dLAem = self.layeratt_d(dGCNem)
            node_embed = mLAem * dLAem
        else:
            node_embed = (xm * xd).squeeze(dim=1)
        pre_part = self.fcLinear(node_embed)
        pre_a = self.sigmoid(pre_part).squeeze(dim=1)
        return pre_a


class EmbeddingM(nn.Module):
    def __init__(self, args):
        super(EmbeddingM, self).__init__()
        self.args = args
        self.gcn_x1_f = GCNConv(self.args.fm, self.args.fm)
        self.gcn_x2_f = GCNConv(self.args.fm, self.args.fm)

        self.gcn_x1_s = GCNConv(self.args.fm, self.args.fm)
        self.gcn_x2_s = GCNConv(self.args.fm, self.args.fm)

        self.gcn_x1_g = GCNConv(self.args.fm, self.args.fm)
        self.gcn_x2_g = GCNConv(self.args.fm, self.args.fm)

        # New RNA-Medicine layers
        self.gcn_md1 = GCNConv(self.args.fm, self.args.fm)
        self.gcn_md2 = GCNConv(self.args.fm, self.args.fm)

        self.fc1_x = nn.Linear(in_features=self.args.view * self.args.gcn_layers1,
                               out_features=5 * self.args.view * self.args.gcn_layers1)
        self.fc2_x = nn.Linear(in_features=5 * self.args.view * self.args.gcn_layers1,
                               out_features=self.args.view * self.args.gcn_layers1)
        self.sigmoidx = nn.Sigmoid()
        self.cnn_x = nn.Conv2d(in_channels=self.args.view * self.args.gcn_layers1, out_channels=1,
                               kernel_size=(1, 1), stride=1, bias=True)

    def forward(self, data, fm1):
        x_m_f1 = torch.relu(self.gcn_x1_f(fm1.cuda(), data['mm_f']['edges'].cuda(), data['mm_f']['data_matrix'][
            data['mm_f']['edges'][0], data['mm_f']['edges'][1]].cuda()))
        x_m_f2 = torch.relu(self.gcn_x2_f(x_m_f1, data['mm_f']['edges'].cuda(), data['mm_f']['data_matrix'][
            data['mm_f']['edges'][0], data['mm_f']['edges'][1]].cuda()))

        x_m_s1 = torch.relu(self.gcn_x1_s(fm1.cuda(), data['mm_s']['edges'].cuda(), data['mm_s']['data_matrix'][
            data['mm_s']['edges'][0], data['mm_s']['edges'][1]].cuda()))
        x_m_s2 = torch.relu(self.gcn_x2_s(x_m_s1, data['mm_s']['edges'].cuda(), data['mm_s']['data_matrix'][
            data['mm_s']['edges'][0], data['mm_s']['edges'][1]].cuda()))

        x_m_g1 = torch.relu(self.gcn_x1_g(fm1.cuda(), data['mm_g']['edges'].cuda(), data['mm_g']['data_matrix'][
            data['mm_g']['edges'][0], data['mm_g']['edges'][1]].cuda()))
        x_m_g2 = torch.relu(self.gcn_x2_g(x_m_g1, data['mm_g']['edges'].cuda(), data['mm_g']['data_matrix'][
            data['mm_g']['edges'][0], data['mm_g']['edges'][1]].cuda()))

        XM = torch.cat((x_m_f1, x_m_f2, x_m_s1, x_m_s2, x_m_g1, x_m_g2), 1).t()
        XM = XM.view(1, self.args.view * self.args.gcn_layers1, self.args.fm, -1)

        globalAvgPool_x = nn.AvgPool2d((self.args.fm, 853), (1, 1))
        x_channel_attention = globalAvgPool_x(XM)
        x_channel_attention = x_channel_attention.view(x_channel_attention.size(0), -1)
        x_channel_attention = self.fc1_x(x_channel_attention)
        x_channel_attention = torch.relu(x_channel_attention)
        x_channel_attention = self.fc2_x(x_channel_attention)
        x_channel_attention = self.sigmoidx(x_channel_attention)
        x_channel_attention = x_channel_attention.view(x_channel_attention.size(0), x_channel_attention.size(1), 1, 1)
        XM_channel_attention = x_channel_attention * XM
        XM_channel_attention = torch.relu(XM_channel_attention)

        x = self.cnn_x(XM_channel_attention)
        x = x.view(self.args.fm, 853).t()
        return x


class EmbeddingD(nn.Module):
    def __init__(self, args):
        super(EmbeddingD, self).__init__()
        self.args = args

        # Existing disease layers
        self.gcn_y1_t = GCNConv(self.args.fd, self.args.fd)
        self.gcn_y2_t = GCNConv(self.args.fd, self.args.fd)
        self.gcn_y1_s = GCNConv(self.args.fd, self.args.fd)
        self.gcn_y2_s = GCNConv(self.args.fd, self.args.fd)
        self.gcn_y1_g = GCNConv(self.args.fd, self.args.fd)
        self.gcn_y2_g = GCNConv(self.args.fd, self.args.fd)

        # New RNA-Disease layers
        self.gcn_rd1 = GCNConv(self.args.fd, self.args.fd)  # Assumes the edge feature is disease-related
        self.gcn_rd2 = GCNConv(self.args.fd, self.args.fd)

        # Other layers remain unchanged
        self.fc1_y = nn.Linear(in_features=self.args.view * self.args.gcn_layers1,
                               out_features=5 * self.args.view * self.args.gcn_layers1)
        self.fc2_y = nn.Linear(in_features=5 * self.args.view * self.args.gcn_layers1,
                               out_features=self.args.view * self.args.gcn_layers1)
        self.sigmoidy = nn.Sigmoid()
        self.cnn_y = nn.Conv2d(in_channels=self.args.view * self.args.gcn_layers1, out_channels=1,
                               kernel_size=(1, 1), stride=1, bias=True)

    def forward(self, data, dm1):
        # Existing Disease forward computations
        y_d_t1 = torch.relu(self.gcn_y1_t(dm1.cuda(), data['dd_t']['edges'].cuda(), data['dd_t']['data_matrix'][
            data['dd_t']['edges'][0], data['dd_t']['edges'][1]].cuda()))
        y_d_t2 = torch.relu(self.gcn_y2_t(y_d_t1, data['dd_t']['edges'].cuda(), data['dd_t']['data_matrix'][
            data['dd_t']['edges'][0], data['dd_t']['edges'][1]].cuda()))
        y_d_s1 = torch.relu(self.gcn_y1_s(dm1.cuda(), data['dd_s']['edges'].cuda(), data['dd_s']['data_matrix'][
            data['dd_s']['edges'][0], data['dd_s']['edges'][1]].cuda()))
        y_d_s2 = torch.relu(self.gcn_y2_s(y_d_s1, data['dd_s']['edges'].cuda(), data['dd_s']['data_matrix'][
            data['dd_s']['edges'][0], data['dd_s']['edges'][1]].cuda()))
        y_d_g1 = torch.relu(self.gcn_y1_g(dm1.cuda(), data['dd_g']['edges'].cuda(), data['dd_g']['data_matrix'][
            data['dd_g']['edges'][0], data['dd_g']['edges'][1]].cuda()))
        y_d_g2 = torch.relu(self.gcn_y2_g(y_d_g1, data['dd_g']['edges'].cuda(), data['dd_g']['data_matrix'][
            data['dd_g']['edges'][0], data['dd_g']['edges'][1]].cuda()))

        # Combining embeddings
        combined_embedding = torch.cat((y_d_t1, y_d_t2, y_d_s1, y_d_s2, y_d_g1, y_d_g2), 1).t()

        combined_embedding = combined_embedding.view(1, self.args.view * self.args.gcn_layers1, self.args.fd, -1)
        globalAvgPool_y = nn.AvgPool2d((self.args.fd, 591), (1, 1))

        y_channel_attention = globalAvgPool_y(combined_embedding)
        y_channel_attention = y_channel_attention.view(y_channel_attention.size(0), -1)
        y_channel_attention = self.fc1_y(y_channel_attention)
        y_channel_attention = torch.relu(y_channel_attention)
        y_channel_attention = self.fc2_y(y_channel_attention)
        y_channel_attention = self.sigmoidy(y_channel_attention)

        y_channel_attention = y_channel_attention.view(y_channel_attention.size(0), y_channel_attention.size(1), 1, 1)
        YD_channel_attention = y_channel_attention * combined_embedding
        YD_channel_attention = torch.relu(YD_channel_attention)

        y = self.cnn_y(YD_channel_attention)
        y = y.view(self.args.fd, 591).t()

        return y


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=32):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size).apply(init),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False).apply(init)
        )

    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)
        return (beta * z).sum(1)


class HANLayer(nn.Module):
    def __init__(self, meta_paths, in_size, out_size, dropout):
        super(HANLayer, self).__init__()
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
        self.gcn_layers = nn.ModuleList()
        for _ in self.meta_paths:
            self.gcn_layers.append(GraphConv(in_size, out_size, activation=F.relu).apply(init))

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(g, meta_path)

        device = h.device

        semantic_embeddings = []
        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            new_g = new_g.to(device)
            semantic_embeddings.append(self.gcn_layers[i](new_g, h))

        return torch.mean(torch.stack(semantic_embeddings, dim=1), dim=1)


class HAN(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, out_size, dropout):
        super(HAN, self).__init__()
        self.layer = HANLayer(meta_paths, in_size, hidden_size, dropout)
        self.predict = nn.Linear(hidden_size, out_size, bias=False).apply(init)

    def forward(self, g, h):
        h = self.layer(g, h)
        return self.predict(h)


class HAN_DTI(nn.Module):
    def __init__(self, all_meta_paths, in_size, hidden_size, out_size, dropout):
        super(HAN_DTI, self).__init__()
        self.sum_layers = nn.ModuleList()
        for i in range(0, len(all_meta_paths)):
            self.sum_layers.append(
                HAN(all_meta_paths[i], in_size, hidden_size, out_size, dropout))

    def forward(self, s_g, s_h_1, s_h_2):
        h_11 = self.sum_layers[0](s_g[0], s_h_1)
        h_21 = self.sum_layers[1](s_g[1], s_h_2)
        return h_11, h_21


def init(i):
    if isinstance(i, nn.Linear):
        torch.nn.init.xavier_uniform_(i.weight)


class MultiHeadAttention(nn.Module):
    def __init__(self, inSize, outSize, gcnlayers, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([LayerAtt2(inSize, outSize) for _ in range(num_heads)])
        self.merge_layer = nn.Linear(num_heads * outSize, outSize)

    def forward(self, x):
        return self.merge_layer(torch.cat([head(x) for head in self.heads], dim=-1))
