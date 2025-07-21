import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

class HGCNLayer(nn.Module):
    """超图卷积层实现"""
    def __init__(self, in_features, out_features):
        super(HGCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x, H, W_e=None):
        """
        参数:
            x: 节点特征矩阵, shape [num_nodes, in_features]
            H: 超图关联矩阵, shape [num_nodes, num_hyperedges]
            W_e: 超边权重矩阵, 默认为单位矩阵
            
        返回:
            更新后的节点特征
        """
        # 如果没有提供超边权重矩阵，则默认为单位矩阵
        if W_e is None:
            W_e = torch.ones(H.shape[1]).diag().to(x.device)
        elif isinstance(W_e, torch.Tensor) and W_e.dim() == 1:
            W_e = torch.diag(W_e).to(x.device)
            
        # 计算节点度矩阵 D_v
        D_v = torch.sparse.sum(H, dim=1).to_dense()
        D_v_sqrt_inv = torch.diag(torch.pow(D_v + 1e-8, -0.5))
        
        # 计算超边度矩阵 D_e
        D_e = torch.sparse.mm(H.t(), H).to_dense()
        D_e_inv = torch.diag(1.0 / (torch.diag(D_e) + 1e-8))
        
        # 特征变换
        support = torch.mm(x, self.weight)
        
        # 超图卷积操作
        # X^(l+1) = σ(D_v^(-1/2) H W D_e^(-1) H^T D_v^(-1/2) X^(l) W^(l))
        output = torch.mm(D_v_sqrt_inv, torch.mm(H, torch.mm(W_e, torch.mm(D_e_inv, torch.mm(H.t(), torch.mm(D_v_sqrt_inv, support))))))
        
        return output

class HGCN(nn.Module):
    """超图卷积神经网络模型"""
    def __init__(self, in_features, hidden_features, out_features, dropout=0.5):
        super(HGCN, self).__init__()
        self.hgc1 = HGCNLayer(in_features, hidden_features)
        self.hgc2 = HGCNLayer(hidden_features, out_features)
        self.dropout = dropout
        
    def forward(self, x, H, W_e=None):
        x = F.relu(self.hgc1(x, H, W_e))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.hgc2(x, H, W_e)
        return F.log_softmax(x, dim=1)

def to_sparse_tensor(indices, values, size):
    """将稀疏矩阵转换为PyTorch稀疏张量"""
    indices = torch.LongTensor(indices).t()
    values = torch.FloatTensor(values)
    return torch.sparse.FloatTensor(indices, values, size)

def construct_hypergraph_incidence(num_nodes, hyperedge_list):
    """构造超图关联矩阵
    
    参数:
        num_nodes: 节点数量
        hyperedge_list: 超边列表，每个超边包含多个节点索引
        
    返回:
        H: 稀疏的超图关联矩阵
    """
    indices = []
    values = []
    
    for e_idx, nodes in enumerate(hyperedge_list):
        for node in nodes:
            indices.append([node, e_idx])
            values.append(1.0)
            
    return to_sparse_tensor(indices, values, [num_nodes, len(hyperedge_list)])

# 测试用例
def test_hgcn():
    # 构造一个简单的超图
    # 超图有5个节点和3个超边
    # 超边1: {0, 1, 2}
    # 超边2: {1, 2, 3}
    # 超边3: {2, 3, 4}
    num_nodes = 5
    num_features = 8
    num_classes = 3
    hyperedge_list = [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
    
    # 构造关联矩阵H
    H = construct_hypergraph_incidence(num_nodes, hyperedge_list)
    
    # 构造节点特征矩阵
    X = torch.randn(num_nodes, num_features)
    
    # 超边权重
    W_e = torch.ones(len(hyperedge_list))
    
    # 创建模型
    model = HGCN(num_features, 16, num_classes)
    
    # 前向传播
    output = model(X, H, W_e)
    
    print("超图节点数:", num_nodes)
    print("超边数:", len(hyperedge_list))
    print("输入特征维度:", num_features)
    print("输出维度:", output.shape)
    print("输出结果:", output)
    
    # 计算损失(假设有标签)
    y = torch.randint(0, num_classes, (num_nodes,))
    loss = F.nll_loss(output, y)
    print("损失值:", loss.item())
    
    return True

def train_hgcn_node_classification():
    # 构造一个更复杂的超图用于节点分类任务
    num_nodes = 100
    num_features = 16
    num_classes = 4
    num_hyperedges = 30
    
    # 随机生成超边
    hyperedge_list = []
    for i in range(num_hyperedges):
        # 每个超边连接3-8个节点
        edge_size = np.random.randint(3, 9)
        nodes = np.random.choice(num_nodes, edge_size, replace=False).tolist()
        hyperedge_list.append(nodes)
    
    # 构造关联矩阵H
    H = construct_hypergraph_incidence(num_nodes, hyperedge_list)
    
    # 生成随机特征和标签
    X = torch.randn(num_nodes, num_features)
    y = torch.randint(0, num_classes, (num_nodes,))
    
    # 划分训练集和测试集
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:int(0.8*num_nodes)] = True
    test_mask = ~train_mask
    
    # 创建模型
    model = HGCN(num_features, 32, num_classes, dropout=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # 训练模型
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(X, H)
        loss = F.nll_loss(output[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            # 评估模型
            model.eval()
            with torch.no_grad():
                output = model(X, H)
                # 计算测试集准确率
                pred = output[test_mask].max(1)[1]
                acc = pred.eq(y[test_mask]).sum().item() / test_mask.sum().item()
                print(f'Epoch {epoch}, Loss: {loss.item():.4f}, Test Acc: {acc:.4f}')
            model.train()
    
    # 最终评估
    model.eval()
    with torch.no_grad():
        output = model(X, H)
        pred = output.max(1)[1]
        train_acc = pred[train_mask].eq(y[train_mask]).sum().item() / train_mask.sum().item()
        test_acc = pred[test_mask].eq(y[test_mask]).sum().item() / test_mask.sum().item()
        print(f'最终结果 - 训练集准确率: {train_acc:.4f}, 测试集准确率: {test_acc:.4f}')
    
    return model, H, X, y

# 运行更复杂的测试用例
if __name__ == "__main__":
    test_hgcn()
    print("\n==== 节点分类任务测试 ====")
    train_hgcn_node_classification()