import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import get_laplacian, to_dense_adj
from torch_geometric.data import Data, Batch
import numpy as np
from torch_scatter import scatter_add

class MultiHeadAttention(nn.Module):
    def __init__(self, in_channels, out_channels, heads=8, dropout=0.0, edge_dim=None, use_edge_features=False):
        super(MultiHeadAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.use_edge_features = use_edge_features
        self.head_dim = out_channels // heads
        
        self.q_lin = nn.Linear(in_channels, out_channels)
        self.k_lin = nn.Linear(in_channels, out_channels)
        self.v_lin = nn.Linear(in_channels, out_channels)
        
        if use_edge_features and edge_dim is not None:
            self.edge_lin = nn.Linear(edge_dim, heads)
            
        self.out_lin = nn.Linear(out_channels, out_channels)
        
    def forward(self, x, edge_index, edge_attr=None):
        # x: [N, F_in], edge_index: [2, E], edge_attr: [E, F_e]
        N = x.size(0)
        
        # 线性变换用于查询、键和值
        q = self.q_lin(x).view(N, self.heads, self.head_dim)  # [N, H, F_out/H]
        k = self.k_lin(x).view(N, self.heads, self.head_dim)  # [N, H, F_out/H]
        v = self.v_lin(x).view(N, self.heads, self.head_dim)  # [N, H, F_out/H]
        
        # 准备注意力计算
        row, col = edge_index  # 源节点和目标节点
        
        # 计算注意力分数
        alpha = (q[row] * k[col]).sum(dim=-1) / np.sqrt(self.head_dim)  # [E, H]
        
        # 如果有边特征，则与注意力分数相乘
        if self.use_edge_features and edge_attr is not None:
            edge_weights = self.edge_lin(edge_attr)  # [E, H]
            alpha = alpha * edge_weights
            
        # 限制注意力分数在 [-5, 5] 范围内
        alpha = torch.clamp(alpha, -5.0, 5.0)
        
        # 对每个节点进行softmax归一化
        alpha = softmax_by_row(alpha, row, num_nodes=x.size(0))
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # 消息传递
        out = scatter_add(v[col].unsqueeze(1) * alpha.unsqueeze(-1), row, dim=0, dim_size=N)  # [N, H, F_out/H]
        out = out.reshape(N, self.out_channels)  # [N, F_out]
        
        # 输出投影
        out = self.out_lin(out)
        
        return out

def softmax_by_row(src, row, num_nodes):
    """按行计算softmax（用于稀疏图中的注意力机制）"""
    # 计算每行的最大值以保持数值稳定性
    max_per_row = scatter_add(src.max(dim=-1, keepdim=True)[0], row, dim=0, dim_size=num_nodes)
    max_per_row = max_per_row[row]
    
    # 计算指数并归一化
    exp_src = torch.exp(src - max_per_row)
    sum_exp = scatter_add(exp_src, row, dim=0, dim_size=num_nodes) + 1e-16
    return exp_src / sum_exp[row]

class GraphTransformerLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=8, dropout=0.0, 
                 edge_dim=None, use_edge_features=False, use_norm=True):
        super(GraphTransformerLayer, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.use_edge_features = use_edge_features
        self.edge_dim = edge_dim
        self.use_norm = use_norm
        
        # 多头注意力
        self.attention = MultiHeadAttention(
            in_channels, out_channels, heads, dropout, edge_dim, use_edge_features
        )
        
        # 前馈网络
        self.ff_linear1 = nn.Linear(out_channels, 2 * out_channels)
        self.ff_linear2 = nn.Linear(2 * out_channels, out_channels)
        
        # 归一化层
        if use_norm:
            self.norm1 = nn.BatchNorm1d(out_channels)
            self.norm2 = nn.BatchNorm1d(out_channels)
            
        # 如果使用边特征，为边特征添加转换
        if use_edge_features and edge_dim is not None:
            self.edge_ff_linear1 = nn.Linear(edge_dim, 2 * edge_dim)
            self.edge_ff_linear2 = nn.Linear(2 * edge_dim, edge_dim)
            if use_norm:
                self.edge_norm1 = nn.BatchNorm1d(edge_dim)
                self.edge_norm2 = nn.BatchNorm1d(edge_dim)
            
    def forward(self, x, edge_index, edge_attr=None):
        """
        x: 节点特征 [N, F_in]
        edge_index: 边索引 [2, E]
        edge_attr: 边特征 [E, F_e]
        """
        # 注意力层
        att_out = self.attention(x, edge_index, edge_attr)
        
        # 第一个残差连接和归一化
        x1 = x + att_out
        if self.use_norm:
            x1 = self.norm1(x1)
        
        # 前馈网络
        x2 = self.ff_linear2(F.relu(self.ff_linear1(x1)))
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        
        # 第二个残差连接和归一化
        x_out = x1 + x2
        if self.use_norm:
            x_out = self.norm2(x_out)
            
        # 如果有边特征，则更新边特征
        edge_out = None
        if self.use_edge_features and edge_attr is not None:
            # 这里简化了边特征的更新，完整实现需要更复杂的边特征交互
            edge1 = edge_attr
            
            if self.use_norm:
                edge1 = self.edge_norm1(edge1)
                
            edge2 = self.edge_ff_linear2(F.relu(self.edge_ff_linear1(edge1)))
            edge2 = F.dropout(edge2, p=self.dropout, training=self.training)
            
            edge_out = edge1 + edge2
            if self.use_norm:
                edge_out = self.edge_norm2(edge_out)
                
        return x_out, edge_out

class LaplacianPE(nn.Module):
    """拉普拉斯位置编码"""
    def __init__(self, max_nodes, embedding_dim, num_eigenvectors=10):
        super(LaplacianPE, self).__init__()
        self.max_nodes = max_nodes
        self.embedding_dim = embedding_dim
        self.num_eigenvectors = num_eigenvectors
        
        # 位置编码投影
        self.pe_encoder = nn.Linear(num_eigenvectors, embedding_dim)
        
    def forward(self, batch):
        """
        计算拉普拉斯位置编码
        batch: PyG Batch对象
        """
        pe_list = []
        
        # 对批次中的每个图计算位置编码
        for i, n_i in enumerate(batch.batch.bincount()):
            # 获取当前图的子图
            edge_index = batch.edge_index[:, batch.batch == i]
            
            # 调整边索引
            edge_index = edge_index - edge_index.min()
            
            # 计算拉普拉斯矩阵特征向量
            L_index, L_weight = get_laplacian(edge_index, normalization='sym', num_nodes=n_i)
            L = to_dense_adj(L_index, edge_attr=L_weight, max_num_nodes=n_i)[0]
            
            # 计算特征值和特征向量
            try:
                eigval, eigvec = torch.linalg.eigh(L)
                # 使用前k个非零特征值对应的特征向量
                pe = eigvec[:, 1:self.num_eigenvectors+1] if n_i > self.num_eigenvectors else torch.zeros(n_i, self.num_eigenvectors, device=batch.x.device)
            except:
                # 如果特征值计算失败，则使用零张量
                pe = torch.zeros(n_i, self.num_eigenvectors, device=batch.x.device)
            
            pe_list.append(pe)
            
        # 合并所有图的位置编码
        pe_batch = torch.cat(pe_list, dim=0)
        
        # 投影位置编码
        pe_projected = self.pe_encoder(pe_batch)
        
        return pe_projected

class GraphTransformer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, heads=8, 
                 dropout=0.0, edge_dim=None, use_edge_features=False, use_pe=True, 
                 max_nodes=1000, num_eigenvectors=10, use_norm=True):
        super(GraphTransformer, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.use_pe = use_pe
        self.use_edge_features = use_edge_features
        
        # 节点特征初始编码
        self.node_encoder = nn.Linear(in_channels, hidden_channels)
        
        # 边特征编码（如果使用）
        if use_edge_features and edge_dim is not None:
            self.edge_encoder = nn.Linear(edge_dim, hidden_channels)
            edge_dim_hidden = hidden_channels
        else:
            edge_dim_hidden = edge_dim
            
        # 拉普拉斯位置编码（如果使用）
        if use_pe:
            self.pe = LaplacianPE(max_nodes, hidden_channels, num_eigenvectors)
            
        # Graph Transformer层
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                GraphTransformerLayer(
                    hidden_channels, hidden_channels, heads, dropout, 
                    edge_dim_hidden, use_edge_features, use_norm
                )
            )
            
        # 输出层
        self.output_layer = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, batch):
        """
        batch: PyG Batch对象，包含:
            - x: 节点特征 [N, F_in]
            - edge_index: 边索引 [2, E]
            - edge_attr: 边特征 [E, F_e]（可选）
            - batch: 批次索引 [N]
        """
        x, edge_index = batch.x, batch.edge_index
        
        # 编码节点特征
        x = self.node_encoder(x)
        
        # 编码边特征（如果存在）
        edge_attr = None
        if self.use_edge_features and hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
            edge_attr = self.edge_encoder(batch.edge_attr)
            
        # 添加位置编码（仅在输入层）
        if self.use_pe:
            pe = self.pe(batch)
            x = x + pe
            
        # 通过Graph Transformer层
        for layer in self.layers:
            x, edge_attr = layer(x, edge_index, edge_attr)
            
        # 输出层
        x = self.output_layer(x)
        
        return x

# 测试用例：图分类任务
def test_graph_transformer():
    # 导入必要的库
    from torch_geometric.datasets import TUDataset
    from torch_geometric.loader import DataLoader
    from torch_geometric.transforms import OneHotDegree
    from sklearn.metrics import accuracy_score
    
    # 设置随机种子
    torch.manual_seed(42)
    
    # 加载数据集（以MUTAG数据集为例）
    dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG', transform=OneHotDegree(10))
    
    # 数据集信息
    print(f"数据集: {dataset.name}")
    print(f"数据集大小: {len(dataset)}")
    print(f"节点特征维度: {dataset.num_node_features}")
    print(f"边特征维度: {dataset.num_edge_features}")
    print(f"类别数量: {dataset.num_classes}")
    
    # 划分数据集
    torch.manual_seed(42)
    dataset = dataset.shuffle()
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 模型参数
    in_channels = dataset.num_node_features
    edge_dim = dataset.num_edge_features if dataset.num_edge_features > 0 else None
    hidden_channels = 64
    out_channels = dataset.num_classes
    use_edge_features = edge_dim is not None and edge_dim > 0
    
    # 创建模型
    model = GraphTransformer(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=3,
        heads=4,
        dropout=0.5,
        edge_dim=edge_dim,
        use_edge_features=use_edge_features,
        use_pe=True,
        max_nodes=100,
        num_eigenvectors=8,
        use_norm=True
    )
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 训练函数
    def train():
        model.train()
        total_loss = 0
        
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data)
            
            # 图级预测：对每个图中的所有节点取平均
            out = scatter_add(out, data.batch, dim=0)
            out = out / torch.bincount(data.batch).unsqueeze(-1)
            
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
            
        return total_loss / len(train_dataset)
    
    # 测试函数
    @torch.no_grad()
    def test(loader):
        model.eval()
        y_true, y_pred = [], []
        
        for data in loader:
            out = model(data)
            
            # 图级预测：对每个图中的所有节点取平均
            out = scatter_add(out, data.batch, dim=0)
            out = out / torch.bincount(data.batch).unsqueeze(-1)
            
            y_true.append(data.y.cpu().numpy())
            y_pred.append(out.argmax(dim=-1).cpu().numpy())
            
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        
        return accuracy_score(y_true, y_pred)
    
    # 训练模型
    print("开始训练...")
    best_acc = 0
    for epoch in range(100):
        loss = train()
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
            
    print(f"最佳测试准确率: {best_acc:.4f}")
    
    return model

if __name__ == "__main__":
    model = test_graph_transformer()