import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import os
from pathlib import Path
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import time
from math import gcd
from functools import reduce

# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 设置CUDA相关参数
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    print(f"当前CUDA设备: {torch.cuda.get_device_name(0)}")
    print(f"可用GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# 添加计时器装饰器
def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"开始执行 {func.__name__}...")
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 执行完成，耗时: {end_time - start_time:.2f} 秒")
        return result
    return wrapper

class SubspaceAwareGNNLayer(nn.Module):
    """
    子空间感知的图神经网络（GNN）层。
    这个自定义层可以为数据中不同的子空间（例如，不同的类别）学习不同的变换权重
    """
    def __init__(self, in_features, out_features, n_subspaces=None):
        super(SubspaceAwareGNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_subspaces = n_subspaces if n_subspaces is not None else 1
        
        # 为每个子空间创建一个权重矩阵
        if self.n_subspaces > 1:
            self.weights = nn.Parameter(torch.FloatTensor(self.n_subspaces, in_features, out_features))
        else:
            self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
            
        self.reset_parameters()
        
    def reset_parameters(self):
        if self.n_subspaces > 1:
            for i in range(self.n_subspaces):
                nn.init.kaiming_uniform_(self.weights[i])
        else:
            nn.init.kaiming_uniform_(self.weight)
        
    def forward(self, input_features, adj, subspace_labels=None):
        """
        前向传播
        
        参数:
        - input_features: 输入特征，形状为 [n_samples, in_features]
        - adj: 邻接矩阵，形状为 [n_samples, n_samples]
        - subspace_labels: 子空间标签，形状为 [n_samples]，值为0到n_subspaces-1
        
        返回:
        - output: 输出特征，形状为 [n_samples, out_features]
        """
        if self.n_subspaces > 1 and subspace_labels is not None:
            # 子空间感知模式
            output = torch.zeros(input_features.shape[0], self.out_features, device=input_features.device)
            
            # 为每个子空间单独处理
            for s in range(self.n_subspaces):
                # 获取当前子空间的样本掩码
                mask = (subspace_labels == s)
                if not torch.any(mask):
                    continue
                    
                # 获取当前子空间的样本
                subspace_features = input_features[mask]
                
                # 应用当前子空间的权重
                support = torch.mm(subspace_features, self.weights[s])
                
                # 修改邻接矩阵，只保留子空间内的连接
                subspace_adj = adj[mask][:, mask]
                
                # 图卷积操作
                subspace_output = torch.spmm(subspace_adj, support)
                
                # 将结果放回原始输出
                output[mask] = subspace_output
                
            return output
        else:
            # 标准GNN模式
            if self.n_subspaces > 1:
                # 使用第一个子空间的权重
                support = torch.mm(input_features, self.weights[0])
            else:
                support = torch.mm(input_features, self.weight)
                
            output = torch.spmm(adj, support)
            return output


class SubspaceAwareGNN(nn.Module):
    """
    子空间感知的图神经网络模型
    """
    def __init__(self, input_dim, hidden_dims=[1024, 512, 256], output_dim=None, n_subspaces=None):
        super(SubspaceAwareGNN, self).__init__()
        self.input_dim = input_dim
        self.n_subspaces = n_subspaces if n_subspaces is not None else 1
        
        if output_dim is None:
            output_dim = input_dim
        
        # 构建GNN层
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(SubspaceAwareGNNLayer(prev_dim, hidden_dim, self.n_subspaces))
            prev_dim = hidden_dim
        
        # 输出层
        self.output_layer = SubspaceAwareGNNLayer(prev_dim, output_dim, self.n_subspaces)
        
        # 预测层 - 为每个子空间创建一个预测器
        if self.n_subspaces > 1:
            self.predictors = nn.ModuleList()
            for _ in range(self.n_subspaces):
                self.predictors.append(nn.Sequential(
                    nn.Linear(output_dim, output_dim),
                    nn.ReLU(),
                    nn.Linear(output_dim, input_dim)
                ))
        else:
            self.predictor = nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, input_dim)
            )
        
    def forward(self, x, adj, subspace_labels=None):
        """
        前向传播
        
        参数:
        - x: 输入特征，形状为 [n_samples, input_dim]
        - adj: 邻接矩阵，形状为 [n_samples, n_samples]
        - subspace_labels: 子空间标签，形状为 [n_samples]，值为0到n_subspaces-1
        
        返回:
        - output: 输出特征，形状为 [n_samples, input_dim]
        """
        # 编码
        h = x
        for layer in self.layers:
            h = F.relu(layer(h, adj, subspace_labels))
            h = F.dropout(h, 0.2, training=self.training)
        
        # 图卷积输出
        h = self.output_layer(h, adj, subspace_labels)
        
        # 预测原始特征
        if self.n_subspaces > 1 and subspace_labels is not None:
            # 为每个子空间使用不同的预测器
            output = torch.zeros_like(x)
            for s in range(self.n_subspaces):
                mask = (subspace_labels == s)
                if torch.any(mask):
                    output[mask] = self.predictors[s](h[mask])
            return output
        else:
            # 使用单一预测器
            if self.n_subspaces > 1:
                return self.predictors[0](h)
            else:
                return self.predictor(h)


def ensure_no_missing(data_filled):
    """确保数据没有缺失值"""
    still_missing = np.isnan(data_filled)
    if np.any(still_missing):
        print(f"填补后仍有 {np.sum(still_missing)} 个缺失值，应用最终填补")
        
        # 复制数据以避免修改原始数据
        result = np.copy(data_filled)
        
        # 对每列单独处理
        for i in range(result.shape[1]):
            col_mask = np.isnan(result[:, i])
            if np.any(col_mask):
                # 尝试使用列均值
                col_mean = np.nanmean(result[:, i])
                if np.isnan(col_mean):  # 整列都是NaN
                    col_mean = 0
                result[col_mask, i] = col_mean
        
        return result
    return data_filled

def build_subspace_aware_graph(data, labels, k=5, metric='cosine', alpha=0.9, similarity_threshold=0.9, batch_size=1000):
    """
    构建子空间感知的图，基于特征相似度加权连接同标签样本
    
    参数:
    - data: 输入数据，形状为 [n_samples, n_features]
    - labels: 标签，形状为 [n_samples]
    - k: 近邻数量
    - metric: 距离度量方式
    - alpha: 子空间内连接的权重系数 (0-1)
    - similarity_threshold: 特征相似度阈值，只有相似度高于此阈值的同标签样本才会连接
    - batch_size: 批处理大小，用于分批计算相似度矩阵以节省内存
    
    返回:
    - adj: 邻接矩阵，形状为 [n_samples, n_samples]
    - knn_indices: 每个样本的KNN索引
    """
    from sklearn.neighbors import kneighbors_graph
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
    
    # 处理缺失值，用于计算距离
    mask = ~np.isnan(data)
    
    # 使用MICE进行填补
    try:
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        import warnings
        warnings.filterwarnings("ignore")
        
        print("为图构建准备数据...")
        # 根据数据维度和子空间特性动态调整
        if data.shape[1] > 1000:  # 高维数据
            n_nearest = min(50, data.shape[1]//20)
        elif data.shape[1] > 100:  # 中等维度
            n_nearest = min(30, data.shape[1]//5)
        else:  # 低维数据
            n_nearest = None  # 使用所有特征

        print(f"使用n_nearest_features={n_nearest}进行MICE填补")
        
        # 创建MICE填补器
        mice_imputer = IterativeImputer(
            max_iter=5000,
            random_state=42,
            verbose=0,
            n_nearest_features=n_nearest
        )
        
        # 使用MICE填补
        print("使用MICE填补数据以构建图...")
        start_time = time.time()
        
        # 检查是否有有效标签，如果有则按子空间进行填补
        if labels is not None and not np.all(np.isnan(labels)):
            print("检测到有效标签，按子空间进行MICE填补...")
            data_filled = np.copy(data)
            
            # 获取有效标签
            valid_mask = ~np.isnan(labels)
            valid_labels = labels[valid_mask]
            
            # 获取唯一标签（子空间）
            unique_labels = np.unique(valid_labels)
            print(f"检测到 {len(unique_labels)} 个子空间")
            
            # 统计每个子空间的样本数
            subspace_counts = {}
            for label in unique_labels:
                count = np.sum(labels == label)
                subspace_counts[label] = count
                print(f"  子空间 {label}: {count} 个样本")
            
            # 计算最佳批次大小的函数
            def calculate_optimal_batch_size(subspace_size):
                """根据子空间大小计算最佳批次大小"""
                if subspace_size <= 100:
                    # 小子空间直接处理
                    return subspace_size
                elif subspace_size <= 200:
                    # 中等子空间使用较大批次
                    return subspace_size
                else:
                    # 大子空间使用子空间大小的因子或固定大小
                    # 确保批次大小是子空间大小的因子，以便完整处理
                    for factor in [2, 3, 4, 5]:
                        if subspace_size % factor == 0:
                            batch = subspace_size // factor
                            if 50 <= batch <= 200:
                                return batch
                    
                    # 如果找不到合适的因子，使用默认值
                    return min(200, max(50, subspace_size // 5))
            
            # 按子空间分别进行MICE填补
            for label in tqdm(unique_labels, desc="按子空间MICE填补"):
                # 获取当前子空间的样本索引
                indices = np.where((labels == label) & valid_mask)[0]
                
                if len(indices) == 0:
                    continue
                
                # 提取当前子空间的数据
                subspace_data = data[indices].copy()
                subspace_mask = mask[indices].copy()
                
                # 检查是否有缺失值
                if np.all(subspace_mask):
                    print(f"  子空间 {label} 没有缺失值，跳过MICE填补")
                    continue
                
                # 先用均值填充获取初始值
                subspace_mean_filled = np.copy(subspace_data)
                for j in range(subspace_data.shape[1]):
                    col_values = subspace_data[:, j][subspace_mask[:, j]]
                    if len(col_values) > 0:
                        col_mean = np.mean(col_values)
                        subspace_mean_filled[:, j] = np.where(subspace_mask[:, j], 
                                                             subspace_data[:, j], 
                                                             col_mean)
                
                try:
                    # 对当前子空间应用MICE
                    subspace_size = len(indices)
                    
                    # 计算最佳批次大小
                    optimal_batch_size = calculate_optimal_batch_size(subspace_size)
                    
                    # 如果子空间样本数较大，需要分批处理
                    if subspace_size > optimal_batch_size:
                        print(f"  子空间 {label} 样本数 {subspace_size}，使用批次大小 {optimal_batch_size}")
                        subspace_filled = np.zeros_like(subspace_data)
                        
                        for j in range(0, subspace_size, optimal_batch_size):
                            end_j = min(j + optimal_batch_size, subspace_size)
                            sub_batch = subspace_mean_filled[j:end_j].copy()
                            sub_batch_mask = subspace_mask[j:end_j]
                            
                            # 只对有缺失值的批次应用MICE
                            if not np.all(sub_batch_mask):
                                try:
                                    sub_filled = mice_imputer.fit_transform(sub_batch)
                                    
                                    # 检查填补后是否仍有缺失值
                                    if np.any(np.isnan(sub_filled)):
                                        print(f"  子空间 {label} 批次 {j//optimal_batch_size + 1} MICE填补后仍有缺失值，使用均值填补")
                                        # 对残留缺失值使用均值填补
                                        for col in range(sub_filled.shape[1]):
                                            col_mask = np.isnan(sub_filled[:, col])
                                            if np.any(col_mask):
                                                col_mean = np.nanmean(sub_filled[:, col])
                                                if np.isnan(col_mean):  # 整列都是NaN
                                                    col_mean = 0
                                                sub_filled[col_mask, col] = col_mean
                                    
                                    subspace_filled[j:end_j] = sub_filled
                                except Exception as e:
                                    print(f"  子空间 {label} 批次 {j//optimal_batch_size + 1} MICE填补失败: {e}")
                                    subspace_filled[j:end_j] = sub_batch  # 使用均值填充的数据
                            else:
                                subspace_filled[j:end_j] = sub_batch
                    else:
                        # 直接处理整个子空间
                        print(f"  子空间 {label} 样本数 {subspace_size}，直接处理")
                        subspace_filled = mice_imputer.fit_transform(subspace_mean_filled)
                        
                        # 检查填补后是否仍有缺失值
                        if np.any(np.isnan(subspace_filled)):
                            print(f"  子空间 {label} MICE填补后仍有缺失值，使用均值填补")
                            # 对残留缺失值使用均值填补
                            for col in range(subspace_filled.shape[1]):
                                col_mask = np.isnan(subspace_filled[:, col])
                                if np.any(col_mask):
                                    col_mean = np.nanmean(subspace_filled[:, col])
                                    if np.isnan(col_mean):  # 整列都是NaN
                                        col_mean = 0
                                    subspace_filled[col_mask, col] = col_mean
                    
                    # 将填补结果放回原数组
                    data_filled[indices] = subspace_filled
                    
                except Exception as e:
                    print(f"  子空间 {label} MICE填补失败: {e}，使用均值填充")
                    data_filled[indices] = subspace_mean_filled
            
        else:
            # 没有有效标签或标签全部缺失，使用原始的分批处理逻辑
            print("没有检测到有效标签，使用常规分批处理进行MICE填补...")
            
            # 检查数据大小，如果太大则分批处理
            n_samples, n_features = data.shape
            if n_samples > 1000:
                print(f"数据较大 ({n_samples} 样本)，使用分批处理进行MICE填补...")
                data_filled = np.zeros_like(data)
                
                # 先用均值填充以获取初始值
                data_mean_filled = np.copy(data)
                for i in range(data.shape[1]):
                    col_mean = np.nanmean(data[:, i])
                    data_mean_filled[:, i] = np.where(mask[:, i], data[:, i], col_mean)
                
                # 计算最佳批次大小
                # 尝试找到能整除样本数的批次大小，以确保完整处理
                batch_size_mice = min(1000, n_samples // 5)  # 默认值
                for factor in [5, 4, 8, 10]:
                    if n_samples % factor == 0:
                        candidate = n_samples // factor
                        if 100 <= candidate <= 1000:
                            batch_size_mice = candidate
                            break
                
                n_batches = (n_samples + batch_size_mice - 1) // batch_size_mice
                print(f"使用批次大小: {batch_size_mice}，共 {n_batches} 批")
                
                for i in tqdm(range(0, n_samples, batch_size_mice), desc="MICE分批填补"):
                    end_idx = min(i + batch_size_mice, n_samples)
                    batch_data = data_mean_filled[i:end_idx].copy()  # 使用均值填充的数据作为起点
                    
                    # 标记当前批次中的缺失值
                    batch_mask = mask[i:end_idx]
                    
                    # 只对有缺失值的批次应用MICE
                    if not np.all(batch_mask):
                        try:
                            batch_filled = mice_imputer.fit_transform(batch_data)
                            
                            # 检查填补后是否仍有缺失值
                            if np.any(np.isnan(batch_filled)):
                                print(f"批次 {i//batch_size_mice + 1}/{n_batches} MICE填补后仍有缺失值，使用均值填补")
                                # 对残留缺失值使用均值填补
                                for col in range(batch_filled.shape[1]):
                                    col_mask = np.isnan(batch_filled[:, col])
                                    if np.any(col_mask):
                                        col_mean = np.nanmean(batch_filled[:, col])
                                        if np.isnan(col_mean):  # 整列都是NaN
                                            col_mean = 0
                                        batch_filled[col_mask, col] = col_mean
                                        
                            data_filled[i:end_idx] = batch_filled
                        except Exception as e:
                            print(f"批次 {i//batch_size_mice + 1}/{n_batches} MICE填补失败: {e}")
                            data_filled[i:end_idx] = batch_data  # 使用均值填充的数据
                    else:
                        data_filled[i:end_idx] = batch_data
            else:
                # 数据量较小，直接处理
                data_filled = mice_imputer.fit_transform(np.where(np.isnan(data), np.nanmean(data, axis=0), data))
                
                # 检查填补后是否仍有缺失值
                if np.any(np.isnan(data_filled)):
                    print("MICE填补后仍有缺失值，使用均值填补")
                    # 对残留缺失值使用均值填补
                    for col in range(data_filled.shape[1]):
                        col_mask = np.isnan(data_filled[:, col])
                        if np.any(col_mask):
                            col_mean = np.nanmean(data_filled[:, col])
                            if np.isnan(col_mean):  # 整列都是NaN
                                col_mean = 0
                            data_filled[col_mask, col] = col_mean
        
        end_time = time.time()
        print(f"MICE填补完成，耗时: {end_time - start_time:.2f} 秒")
        
    except (ImportError, ValueError) as e:
        # 回退到均值填补
        print(f"MICE方法失败，错误: {e}，回退到均值填补")
        print("使用均值填补数据以构建图...")
        data_filled = np.copy(data)
        for i in tqdm(range(data.shape[1]), desc="均值填补"):
            col_mean = np.nanmean(data[:, i])
            data_filled[:, i] = np.where(mask[:, i], data[:, i], col_mean)
    
    # 确保没有残留的缺失值
    data_filled = ensure_no_missing(data_filled)
    
    print("构建KNN图...")
    start_time = time.time()
    
    # 如果样本数量大于5000，使用近似最近邻算法
    n_samples = data.shape[0]
    if n_samples > 5000 and torch.cuda.is_available():
        print(f"样本数量较大 ({n_samples})，使用GPU加速的KNN...")
        try:
            # 使用GPU加速KNN计算
            data_tensor = torch.FloatTensor(data_filled).to(device)
            adj = torch.zeros((n_samples, n_samples), device=device)
            
            # 使用子空间感知的批处理大小
            batch_size_knn = calculate_subspace_aware_batch_size(labels, default_size=360)
            print(f"使用子空间感知的批处理大小: {batch_size_knn}")
            
            for i in tqdm(range(0, n_samples, batch_size_knn), desc="计算KNN (GPU)"):
                end_i = min(i + batch_size_knn, n_samples)
                batch_i = data_tensor[i:end_i]
                
                # 计算当前批次与所有样本的距离
                if metric == 'cosine':
                    # 计算余弦相似度
                    batch_norm = torch.norm(batch_i, dim=1, keepdim=True)
                    all_norm = torch.norm(data_tensor, dim=1, keepdim=True)
                    
                    # 避免除以零
                    batch_norm[batch_norm == 0] = 1e-8
                    all_norm[all_norm == 0] = 1e-8
                    
                    batch_normalized = batch_i / batch_norm
                    all_normalized = data_tensor / all_norm
                    
                    # 计算相似度矩阵 (越大越相似)
                    sim_matrix = torch.mm(batch_normalized, all_normalized.t())
                    
                    # 对每行找出最大的k个值（不包括自身）
                    for j in range(i, end_i):
                        row_idx = j - i
                        sim_row = sim_matrix[row_idx]
                        sim_row[j] = -1  # 排除自身
                        
                        # 找出前k个最大值的索引
                        _, topk_indices = torch.topk(sim_row, k)
                        
                        # 设置邻接矩阵
                        adj[j, topk_indices] = 1
                else:
                    # 欧氏距离 (越小越相似)
                    for j in range(i, end_i):
                        row_idx = j - i
                        diff = data_tensor[j:j+1] - data_tensor
                        dist = torch.sum(diff * diff, dim=1)
                        
                        # 排除自身
                        dist[j] = float('inf')
                        
                        # 找出前k个最小值的索引
                        _, topk_indices = torch.topk(dist, k, largest=False)
                        
                        # 设置邻接矩阵
                        adj[j, topk_indices] = 1
            
            # 转换为CPU并转为numpy数组
            adj = adj.cpu().numpy()
            
        except Exception as e:
            print(f"GPU加速KNN失败: {e}，回退到CPU计算")
            # 回退到sklearn的kneighbors_graph
            knn = kneighbors_graph(data_filled, k, metric=metric, include_self=False)
            adj = knn.toarray()
    else:
        # 使用sklearn的kneighbors_graph
        knn = kneighbors_graph(data_filled, k, metric=metric, include_self=False)
        adj = knn.toarray()
    
    end_time = time.time()
    print(f"KNN图构建完成，耗时: {end_time - start_time:.2f} 秒")
    
    # 获取每个样本的KNN索引
    n_samples = data.shape[0]
    knn_indices = []
    for i in range(n_samples):
        neighbors = np.nonzero(adj[i])[0]
        knn_indices.append(neighbors)
    
    # 如果有标签信息，增强子空间内的连接
    if labels is not None:
        print("增强子空间内的连接...")
        
        # 计算所有样本对之间的特征相似度 - 使用批处理以节省内存
        n_samples = data.shape[0]
        
        # 找出有效标签
        valid_mask = ~np.isnan(labels)
        valid_indices = np.where(valid_mask)[0]
        n_valid = len(valid_indices)
        
        print(f"计算 {n_valid} 个有效样本的相似度矩阵...")
        
        # 创建标签掩码矩阵 - 使用稀疏矩阵以节省内存
        from scipy.sparse import lil_matrix
        label_mask = lil_matrix((n_samples, n_samples), dtype=np.float32)
        
        # 预处理标签，创建标签到索引的映射
        label_to_indices = {}
        for i, idx in enumerate(valid_indices):
            lbl = labels[idx]
            if lbl not in label_to_indices:
                label_to_indices[lbl] = []
            label_to_indices[lbl].append(idx)
        
        print(f"处理 {len(label_to_indices)} 个不同标签的样本组...")
        
        # 对每个标签组单独处理
        for label, indices in tqdm(label_to_indices.items(), desc="处理标签组"):
            n_indices = len(indices)
            
            # 计算最佳批次大小
            optimal_batch_size = min(batch_size, max(72, n_indices // 2))
            
            # 如果同一标签的样本数量太多，分批处理
            if n_indices > optimal_batch_size:
                for i in range(0, n_indices, optimal_batch_size):
                    batch_indices = indices[i:min(i+optimal_batch_size, n_indices)]
                    process_batch(batch_indices, indices, data_filled, label_mask, 
                                 similarity_threshold, metric, device)
            else:
                # 直接处理
                process_batch(indices, indices, data_filled, label_mask, 
                             similarity_threshold, metric, device)
        
        print(f"处理 {len(label_to_indices)} 个不同标签的样本组...")
        
        # 对每个标签组单独处理
        for label, indices in tqdm(label_to_indices.items(), desc="处理标签组"):
            n_indices = len(indices)
            
            # 计算子空间感知的批处理大小
            # 对于同一标签的样本，我们希望批次大小是样本数量的因子，以避免分割
            # 尝试找到一个接近但不超过原始批次大小的因子
            factors = []
            for i in range(1, int(np.sqrt(n_indices)) + 1):
                if n_indices % i == 0:
                    factors.append(i)
                    if i != n_indices // i:
                        factors.append(n_indices // i)
            factors.sort()
            
            # 选择最接近但不超过batch_size的最大因子
            optimal_batch_size = 1  # 默认值
            for factor in factors:
                if factor <= batch_size:
                    optimal_batch_size = factor
                else:
                    break
            
            # 如果最大因子太小（小于原始批次大小的20%），则使用原始批次大小
            if optimal_batch_size < batch_size * 0.2:
                optimal_batch_size = min(batch_size, n_indices)
            
            print(f"  标签 {label} 的样本数: {n_indices}，使用批次大小: {optimal_batch_size} (因子: {n_indices % optimal_batch_size == 0})")
            
            # 如果同一标签的样本数量太多，分批处理
            if n_indices > optimal_batch_size:
                for i in range(0, n_indices, optimal_batch_size):
                    batch_indices = indices[i:min(i+optimal_batch_size, n_indices)]
                    process_batch(batch_indices, indices, data_filled, label_mask, 
                                 similarity_threshold, metric, device)
            else:
                # 直接处理
                process_batch(indices, indices, data_filled, label_mask, 
                             similarity_threshold, metric, device)
        
        
        
        
        print("合并KNN图和子空间连接...")
        # 将稀疏矩阵转换为密集矩阵
        label_mask = label_mask.toarray()
        
        # 增强子空间内的连接，使用加权平均
        adj = alpha * adj + (1 - alpha) * label_mask
    
    # 确保对称性
    adj = np.maximum(adj, adj.T)
    
    print("转换为PyTorch张量并归一化...")
    # 转换为PyTorch张量
    adj_tensor = torch.FloatTensor(adj)
    
    # 添加自环
    adj_tensor = adj_tensor + torch.eye(adj_tensor.shape[0])
    
    # 度矩阵
    D = torch.diag(torch.sum(adj_tensor, dim=1))
    
    # 归一化邻接矩阵: D^(-1/2) * A * D^(-1/2)
    D_inv_sqrt = torch.pow(D, -0.5)
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0
    adj_normalized = torch.mm(torch.mm(D_inv_sqrt, adj_tensor), D_inv_sqrt)
    
    return adj_normalized, knn_indices

def process_batch(batch_indices, all_indices, data_filled, label_mask, similarity_threshold, metric, device):
    """
    处理一批样本与所有同标签样本之间的相似度计算
    
    参数:
    - batch_indices: 当前批次的样本索引
    - all_indices: 所有同标签的样本索引
    - data_filled: 填充后的数据
    - label_mask: 标签掩码矩阵 (稀疏矩阵)
    - similarity_threshold: 相似度阈值
    - metric: 距离度量方式
    - device: 计算设备
    """
    batch_data = data_filled[batch_indices]
    all_data = data_filled[all_indices]
    
    # 使用GPU加速相似度计算
    if torch.cuda.is_available():
        batch_tensor = torch.FloatTensor(batch_data).to(device)
        all_tensor = torch.FloatTensor(all_data).to(device)
        
        if metric == 'cosine':
            # 计算余弦相似度
            batch_norm = torch.norm(batch_tensor, dim=1, keepdim=True)
            all_norm = torch.norm(all_tensor, dim=1, keepdim=True)
            
            # 避免除以零
            batch_norm[batch_norm == 0] = 1e-8
            all_norm[all_norm == 0] = 1e-8
            
            batch_normalized = batch_tensor / batch_norm
            all_normalized = all_tensor / all_norm
            
            # 计算相似度矩阵
            batch_sim = torch.mm(batch_normalized, all_normalized.t()).cpu().numpy()
            
            # 将相似度归一化到[0, 1]
            batch_sim = (batch_sim + 1) / 2
        else:
            # 欧氏距离
            batch_squared = torch.sum(batch_tensor ** 2, dim=1, keepdim=True)
            all_squared = torch.sum(all_tensor ** 2, dim=1, keepdim=True)
            
            cross_term = torch.mm(batch_tensor, all_tensor.t())
            dist_matrix = batch_squared + all_squared.t() - 2 * cross_term
            dist_matrix = torch.clamp(dist_matrix, min=0)  # 避免数值误差导致的负值
            dist_matrix = torch.sqrt(dist_matrix).cpu().numpy()
            
            # 将距离转换为相似度
            max_dist = np.max(dist_matrix) if np.max(dist_matrix) > 0 else 1.0
            batch_sim = 1 - (dist_matrix / max_dist)
    else:
        # 回退到CPU计算
        if metric == 'cosine':
            batch_sim = cosine_similarity(batch_data, all_data)
            batch_sim = (batch_sim + 1) / 2
        else:
            dist_matrix = euclidean_distances(batch_data, all_data)
            max_dist = np.max(dist_matrix) if np.max(dist_matrix) > 0 else 1.0
            batch_sim = 1 - (dist_matrix / max_dist)
    
    # 构建基于特征相似度的标签掩码矩阵
    for idx, orig_idx in enumerate(batch_indices):
        for j_idx, j in enumerate(all_indices):
            if orig_idx != j:  # 不连接自身
                # 只有当特征相似度高于阈值时才连接
                sim = batch_sim[idx, j_idx]
                if sim >= similarity_threshold:
                    # 使用相似度作为连接权重
                    label_mask[orig_idx, j] = sim

def masked_mse_loss(output, target, mask):
    """
    计算带掩码的MSE损失
    """
    diff = (output - target) * mask
    return torch.sum(diff ** 2) / torch.sum(mask)


def enhanced_masked_loss(output, target, mask, subspace_labels=None, feature_weights=None, 
                         use_ssim=True, img_size=32, alpha_mse=0.7, alpha_ssim=0.3, 
                         alpha_subspace=1):
    """
    增强版损失函数，结合MSE、结构相似性和权重机制
    
    参数:
    - output: 模型输出
    - target: 目标值
    - mask: 掩码矩阵(1表示观测值，0表示缺失值)
    - subspace_labels: 样本所属子空间标签
    - feature_weights: 特征权重（不再使用）
    - use_ssim: 是否使用结构相似性损失
    - img_size: 图像大小（用于SSIM计算）
    - alpha_mse: MSE损失权重
    - alpha_ssim: SSIM损失权重
    - alpha_subspace: 子空间权重系数
    - alpha_feature: 特征权重系数
    
    返回:
    - loss: 综合损失值
    """
    n_samples = output.shape[0]
    n_features = output.shape[1]
    device = output.device
    
    # 基础MSE损失
    diff = (output - target) * mask
    
    # 1. 应用子空间权重
    if subspace_labels is not None:
        # 计算每个子空间的平均误差
        unique_subspaces = torch.unique(subspace_labels)
        n_subspaces = len(unique_subspaces)
        
        # 创建子空间权重矩阵
        subspace_weights = torch.ones(n_samples, device=device)
        
        # 计算每个子空间的平均损失
        subspace_losses = []
        for s in unique_subspaces:
            s_mask = (subspace_labels == s)
            if not torch.any(s_mask):
                continue
                
            # 计算当前子空间的平均损失
            s_diff = diff[s_mask]
            s_mask_vals = mask[s_mask]
            if torch.sum(s_mask_vals) > 0:
                s_loss = torch.sum(s_diff ** 2) / torch.sum(s_mask_vals)
                subspace_losses.append((s, s_loss))
        
        # 根据损失大小分配权重 - 损失越大权重越大（更关注难填补的子空间）
        if len(subspace_losses) > 1:
            # 归一化子空间损失
            s_losses = torch.tensor([l for _, l in subspace_losses], device=device)
            min_loss = torch.min(s_losses)
            max_loss = torch.max(s_losses)
            
            # 避免除以零
            if min_loss != max_loss:
                norm_losses = (s_losses - min_loss) / (max_loss - min_loss)
                
                # 将归一化损失转换为权重
                s_weights = 1.0 + norm_losses  # 确保权重至少为1
                
                # 应用权重到对应子空间
                for i, (s, _) in enumerate(subspace_losses):
                    subspace_weights[subspace_labels == s] = s_weights[i]
        
        # 应用子空间权重到差值
        diff = diff * subspace_weights.view(-1, 1)
    
    # 2. 不再使用特征权重，但保留缺失率计算
    # 计算特征的缺失率
    feature_missing_rate = 1 - torch.mean(mask, dim=0)
    
    # 计算最终的MSE损失
    mse_loss = torch.sum(diff ** 2) / torch.sum(mask)
    
    # 3. 结构相似性损失 (SSIM)
    if use_ssim and n_features == img_size * img_size:  # 只对图像数据使用SSIM
        try:
            ssim_loss = 0
            valid_samples = 0
            
            # 重塑为图像格式进行SSIM计算
            output_images = output.view(-1, 1, img_size, img_size)  # [B, C, H, W]
            target_images = target.view(-1, 1, img_size, img_size)
            mask_images = mask.view(-1, 1, img_size, img_size)
            
            # 逐样本计算SSIM
            for i in range(n_samples):
                # 只对掩码覆盖率超过50%的样本计算SSIM
                if torch.mean(mask_images[i]) > 0.5:
                    # 使用PyTorch的SSIM实现或自定义实现
                    sample_ssim = 1 - compute_ssim(output_images[i], target_images[i], mask_images[i])
                    ssim_loss += sample_ssim
                    valid_samples += 1
            
            # 平均SSIM损失
            if valid_samples > 0:
                ssim_loss = ssim_loss / valid_samples
            else:
                ssim_loss = torch.tensor(0.0, device=device)
                
            # 组合MSE和SSIM损失
            combined_loss = alpha_mse * mse_loss + alpha_ssim * ssim_loss
            return combined_loss
            
        except Exception as e:
            print(f"SSIM计算错误: {e}，回退到MSE损失")
            return mse_loss
    
    return mse_loss


def compute_ssim(img1, img2, mask=None, window_size=11, sigma=1.5):
    """
    计算带掩码的结构相似性指数(SSIM)
    
    参数:
    - img1, img2: 输入图像，形状为 [C, H, W]
    - mask: 掩码图像，形状为 [C, H, W]
    - window_size: 高斯窗口大小
    - sigma: 高斯窗口标准差
    
    返回:
    - ssim_value: SSIM值，范围为[0,1]，1表示完全相同
    """
    # 检查设备
    device = img1.device
    
    # 创建高斯窗口
    window = create_gaussian_window(window_size, sigma).to(device)
    
    # 如果有掩码，应用掩码
    if mask is not None:
        img1 = img1 * mask
        img2 = img2 * mask
    
    # 计算均值
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=1)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=1)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # 计算方差和协方差
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=1) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=1) - mu1_mu2
    
    # SSIM稳定常数
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # 计算SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    # 如果有掩码，只考虑掩码覆盖区域
    if mask is not None:
        # 下采样掩码以匹配SSIM图
        mask_downsampled = F.conv2d(mask, window, padding=window_size//2, groups=1)
        mask_downsampled = (mask_downsampled > 0.5).float()  # 二值化
        
        # 计算掩码区域的平均SSIM
        ssim_masked = (ssim_map * mask_downsampled).sum() / (mask_downsampled.sum() + 1e-8)
        return ssim_masked
    
    return ssim_map.mean()


def create_gaussian_window(window_size, sigma):
    """
    创建高斯窗口用于SSIM计算
    
    参数:
    - window_size: 窗口大小
    - sigma: 高斯标准差
    
    返回:
    - window: 高斯窗口，形状为 [1, 1, window_size, window_size]
    """
    coords = torch.arange(window_size, dtype=torch.float)
    coords -= window_size // 2
    
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    
    window = g.unsqueeze(0) * g.unsqueeze(1)
    window = window.unsqueeze(0).unsqueeze(0)
    
    return window


def calculate_subspace_aware_batch_size(labels, default_size=360, min_multiplier=3, max_multiplier=10):
    """
    计算子空间感知的批处理大小，确保批次大小是子空间样本数量的整数倍
    
    参数:
    - labels: 标签数组
    - default_size: 默认批处理大小
    - min_multiplier: 最小倍数
    - max_multiplier: 最大倍数
    
    返回:
    - batch_size: 优化后的批处理大小
    """
    # 检查是否有有效标签
    valid_mask = ~np.isnan(labels)
    if not np.any(valid_mask):
        return default_size  # 没有有效标签，使用默认大小
    
    # 获取唯一标签
    unique_labels = np.unique(labels[valid_mask])
    
    # 计算每个子空间的样本数量
    subspace_sizes = []
    for label in unique_labels:
        size = np.sum(labels == label)
        subspace_sizes.append(size)
    
    # 如果只有一个子空间，直接返回该子空间大小的整数倍
    if len(subspace_sizes) == 1:
        subspace_size = subspace_sizes[0]
        # 选择合适的倍数
        for multiplier in range(min_multiplier, max_multiplier + 1):
            if subspace_size * multiplier <= default_size * 1.5:
                return subspace_size * multiplier
        # 如果所有倍数都太大，返回子空间大小
        return subspace_size
    
    # 找出最大的子空间大小
    max_subspace_size = max(subspace_sizes)
    
    # 找出最小的子空间大小
    min_subspace_size = min(subspace_sizes)
    
    # 尝试找到一个能被所有子空间大小整除的批次大小
    # 首先尝试使用最大公约数(GCD)的倍数
    
    def find_gcd(numbers):
        return reduce(gcd, numbers)
    
    common_divisor = find_gcd(subspace_sizes)
    
    # 如果公约数太小(小于10)，可能不够有效
    if common_divisor >= 10:
        # 选择公约数的倍数，使其接近但不超过默认大小
        multiplier = default_size // common_divisor
        if multiplier < min_multiplier:
            multiplier = min_multiplier
        return common_divisor * multiplier
    
    # 如果没有合适的公约数，尝试找到一个能被最大子空间大小整除的批次大小
    # 确保批次大小不会太小或太大
    for multiplier in range(min_multiplier, max_multiplier + 1):
        batch_size = max_subspace_size * multiplier
        if batch_size >= default_size * 0.7 and batch_size <= default_size * 1.5:
            return batch_size
    
    # 如果以上方法都不适用，返回默认大小的整数倍，使其大于最大子空间大小
    return ((max_subspace_size + default_size - 1) // default_size) * default_size


# 在subspace_aware_imputation函数中更新相关代码，添加增强的损失函数
@timer
def subspace_aware_imputation(data, labels, mask=None, n_subspaces=None, epochs=200, lr=0.001, k=5, hidden_dims=[1024, 512, 256], alpha=0.9, verbose=True, batch_size=1000, use_enhanced_loss=True, img_size=32):
    """
    使用子空间感知的图神经网络进行缺失值填补
    
    参数:
    - data: 输入数据，形状为 [n_samples, n_features]
    - labels: 标签，形状为 [n_samples]
    - mask: 掩码，True表示观测值，False表示缺失值
    - n_subspaces: 子空间数量，如果为None则自动从标签中推断
    - epochs: 训练轮数
    - lr: 学习率
    - k: KNN图中的近邻数量
    - hidden_dims: GNN隐藏层维度
    - alpha: 子空间内连接的权重系数 (0-1)
    - verbose: 是否显示进度条
    - batch_size: 批处理大小，用于分批计算相似度矩阵以节省内存
    - use_enhanced_loss: 是否使用增强版损失函数
    - img_size: 图像大小，用于SSIM计算
    
    返回:
    - imputed_data: 填补后的数据
    """
    if mask is None:
        mask = ~np.isnan(data)
    
    # 数据准备
    print("准备数据...")
    data_filled, _ = prepare_data_for_imputation(data, mask, labels)
    
    # 处理标签
    print("处理标签...")
    valid_labels = ~np.isnan(labels)
    if np.any(valid_labels):
        # 将标签转换为整数类别
        unique_labels = np.unique(labels[valid_labels])
        label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
        
        # 创建子空间标签数组
        subspace_labels = np.zeros(labels.shape, dtype=int)
        for i, lbl in enumerate(labels):
            if not np.isnan(lbl):
                subspace_labels[i] = label_map[lbl]
            else:
                # 对于未知标签，暂时分配到子空间0
                subspace_labels[i] = 0
        
        # 确定子空间数量
        if n_subspaces is None:
            n_subspaces = len(unique_labels)
    else:
        # 如果没有有效标签，则使用单一子空间
        subspace_labels = np.zeros(labels.shape, dtype=int)
        if n_subspaces is None:
            n_subspaces = 1
    
    print(f"检测到 {n_subspaces} 个子空间")
    
    # 构建子空间感知的图
    adj, knn_indices = build_subspace_aware_graph(data_filled, labels, k=k, alpha=alpha, batch_size=batch_size)
    
    # 标准化
    print("标准化数据...")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_filled)
    
    # 转换为PyTorch张量
    print("转换数据为PyTorch张量...")
    data_tensor = torch.FloatTensor(data_scaled).to(device)
    mask_tensor = torch.FloatTensor(mask.astype(float)).to(device)
    adj_tensor = adj.to(device)
    subspace_tensor = torch.LongTensor(subspace_labels).to(device)
    
    # 不再计算特征权重
    feature_weights = None
    
    # 创建模型
    print("创建GNN模型...")
    model = SubspaceAwareGNN(
        input_dim=data.shape[1], 
        hidden_dims=hidden_dims, 
        n_subspaces=n_subspaces
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 训练模型
    model.train()
    pbar = tqdm(range(epochs)) if verbose else range(epochs)
    losses = []
    
    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=verbose)
    
    print("开始训练GNN模型...")
    train_start_time = time.time()
    
    for epoch in pbar:
        # 前向传播
        optimizer.zero_grad()
        output = model(data_tensor, adj_tensor, subspace_tensor)
        
        # 计算损失
        if use_enhanced_loss:
            loss = enhanced_masked_loss(
                output=output, 
                target=data_tensor, 
                mask=mask_tensor,
                subspace_labels=subspace_tensor,
                feature_weights=None,  # 不再使用特征权重
                use_ssim=(data.shape[1] == img_size * img_size),  # 只对图像使用SSIM
                img_size=img_size
            )
        else:
            loss = masked_mse_loss(output, data_tensor, mask_tensor)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 更新学习率
        scheduler.step(loss)
        
        # 更新进度条
        losses.append(loss.item())
        if verbose:
            pbar.set_description(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
        
        # 更新填补值
        with torch.no_grad():
            model.eval()
            output = model(data_tensor, adj_tensor, subspace_tensor)
            # 只更新缺失值
            data_tensor = data_tensor * mask_tensor + output * (1 - mask_tensor)
            model.train()
    
    train_end_time = time.time()
    print(f"GNN模型训练完成，耗时: {train_end_time - train_start_time:.2f} 秒")
    
    # 最终预测
    print("生成最终预测...")
    model.eval()
    with torch.no_grad():
        output = model(data_tensor, adj_tensor, subspace_tensor)
        # 只更新缺失值
        final_output = data_tensor * mask_tensor + output * (1 - mask_tensor)
    
    # 反标准化
    print("反标准化数据...")
    imputed_data_scaled = final_output.cpu().numpy()
    imputed_data = scaler.inverse_transform(imputed_data_scaled)
    
    # 只替换缺失值
    result = np.copy(data)
    result[~mask] = imputed_data[~mask]
    
    # 可视化损失曲线
    if verbose:
        print("生成损失曲线...")
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('subspace_aware_gnn_loss.png')
        plt.close()
    
    return result


def prepare_data_for_imputation(data, mask=None, labels=None, img_size=32):
    """
    准备用于填补的数据，考虑图像的空间结构，使用MICE方法进行填补
    
    参数:
    - data: 输入数据，形状为 [n_samples, n_features]
    - mask: 掩码，True表示观测值，False表示缺失值
    - labels: 样本标签，用于按类别计算均值
    - img_size: 图像大小，默认为32x32
    
    返回:
    - data_filled: 填充后的数据
    - mask: 掩码
    """
    if mask is None:
        mask = ~np.isnan(data)
    
    data_filled = np.copy(data)
    n_samples = data.shape[0]
    n_features = data.shape[1]
    
    # 检查是否为方形图像
    if img_size * img_size != n_features:
        print(f"警告: 特征数量 {n_features} 不等于 {img_size}x{img_size}={img_size*img_size}，使用传统填补方法")
        return prepare_data_for_imputation_traditional(data, mask, labels)
    
    # 将数据重塑为图像形状以便空间填补
    data_reshaped = data.reshape(n_samples, img_size, img_size)
    mask_reshaped = mask.reshape(n_samples, img_size, img_size)
    data_filled_reshaped = np.copy(data_reshaped)
    
    try:
        # 尝试导入MICE相关库
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        from sklearn.linear_model import BayesianRidge
        
        print("使用MICE方法进行图像数据填补")
        
        # 对每个样本单独处理
        print(f"开始处理 {n_samples} 个样本的MICE填补...")
        start_time = time.time()
        
        for idx in tqdm(range(n_samples), desc="MICE填补进度"):
            # 获取当前图像
            img = data_reshaped[idx]
            img_mask = mask_reshaped[idx]
            
            # 如果图像没有缺失值，则跳过
            if np.all(img_mask):
                continue
                
            
            
            # 将图像重塑为2D矩阵，每行是一个像素，每列是一个特征
            # 这里我们将图像视为一组行向量
            img_2d = img.reshape(img_size*img_size, 1)
            img_mask_2d = img_mask.reshape(img_size*img_size, 1)
            
            # 创建MICE估计器
            # 使用BayesianRidge作为基础估计器，这对于小数据集效果较好
            estimator = BayesianRidge()
            imputer = IterativeImputer(
                estimator=estimator,
                max_iter=5000,
                random_state=42,
                verbose=0,
                n_nearest_features=min(300, img_size),  # 使用最近的10个特征或全部特征
                skip_complete=True  # 跳过没有缺失值的列
            )
            
            # 准备用于MICE的数据
            # 为了增加特征维度，我们添加像素的位置信息
            rows, cols = np.mgrid[0:img_size, 0:img_size]
            positions = np.column_stack([rows.ravel(), cols.ravel()])
            
            # 组合位置信息和像素值
            features = np.column_stack([positions, img_2d])
            
            # 创建缺失值掩码
            missing_mask = ~img_mask_2d.ravel()
            
            # 如果缺失值比例太高，MICE可能效果不佳，先进行简单填充
            if np.mean(missing_mask) > 0.5:
                # 使用列均值进行初步填充
                for j in range(img_size):
                    valid_values = img[:, j][img_mask[:, j]]
                    if len(valid_values) > 0:
                        column_mean = np.mean(valid_values)
                        missing_rows = np.where(~img_mask[:, j])[0]
                        for i in missing_rows:
                            features[i*img_size+j, 2] = column_mean
            
            # 应用MICE
            imputed_features = imputer.fit_transform(features)
            
            # 提取填充后的像素值并重塑回图像
            imputed_img = imputed_features[:, 2].reshape(img_size, img_size)
            
            # 只替换缺失值
            data_filled_reshaped[idx][~img_mask] = imputed_img[~img_mask]
            
        end_time = time.time()
        print(f"MICE填补完成，耗时: {end_time - start_time:.2f} 秒")
        
    except (ImportError, ValueError) as e:
        print(f"MICE方法失败，错误: {e}，回退到传统填补方法")
        
        # 回退到传统的填补策略
        # 对每个样本单独处理
        for idx in tqdm(range(n_samples), desc="传统填补进度"):
            # 对于图像中的每一列
            for j in range(img_size):
                # 获取当前列的有效值
                valid_values = data_reshaped[idx, :, j][mask_reshaped[idx, :, j]]
                
                # 如果有有效值，计算列均值并填补该列中的缺失值
                if len(valid_values) > 0:
                    column_mean = np.mean(valid_values)
                    # 找出该列中的缺失位置
                    missing_rows = np.where(~mask_reshaped[idx, :, j])[0]
                    # 填补缺失值
                    for i in missing_rows:
                        data_filled_reshaped[idx, i, j] = column_mean
        
        # 对于仍然缺失的值，尝试使用行均值填补
        still_missing = ~mask_reshaped & np.isnan(data_filled_reshaped)
        if np.any(still_missing):
            print("部分缺失值无法通过列均值填补，尝试使用行均值填补")
            for idx in tqdm(range(n_samples), desc="行均值填补"):
                # 对于图像中的每一行
                for i in range(img_size):
                    # 获取当前行的有效值
                    valid_values = data_reshaped[idx, i, :][mask_reshaped[idx, i, :]]
                    
                    # 如果有有效值，计算行均值并填补该行中的缺失值
                    if len(valid_values) > 0:
                        row_mean = np.mean(valid_values)
                        # 找出该行中的缺失位置
                        missing_cols = np.where(~mask_reshaped[idx, i, :] & np.isnan(data_filled_reshaped[idx, i, :]))[0]
                        # 填补缺失值
                        for j in missing_cols:
                            data_filled_reshaped[idx, i, j] = row_mean
        
        # 对于仍然缺失的值，使用图像整体均值填补
        still_missing = ~mask_reshaped & np.isnan(data_filled_reshaped)
        if np.any(still_missing):
            print("部分缺失值无法通过行/列均值填补，使用图像整体均值填补")
            for idx in tqdm(range(n_samples), desc="整体均值填补"):
                # 获取当前图像的有效值
                valid_values = data_reshaped[idx][mask_reshaped[idx]]
                
                # 如果有有效值，计算整体均值并填补剩余缺失值
                if len(valid_values) > 0:
                    image_mean = np.mean(valid_values)
                    # 找出该图像中仍然缺失的位置
                    missing_pixels = still_missing[idx]
                    # 填补缺失值
                    data_filled_reshaped[idx][missing_pixels] = image_mean
                else:
                    # 如果整个图像都没有有效值，使用所有图像的均值
                    all_valid_values = data_reshaped[mask_reshaped]
                    if len(all_valid_values) > 0:
                        global_mean = np.mean(all_valid_values)
                        data_filled_reshaped[idx][missing_pixels] = global_mean
    
    # 检查是否仍有缺失值
    still_missing = np.isnan(data_filled_reshaped)
    if np.any(still_missing):
        print("仍有缺失值，使用全局均值填充")
        global_mean = np.nanmean(data_reshaped)
        data_filled_reshaped[still_missing] = global_mean
    
    # 将填补后的数据重新展平
    data_filled = data_filled_reshaped.reshape(n_samples, n_features)
    print("数据填补完成，准备构建图...")
    
    return data_filled, mask


def prepare_data_for_imputation_traditional(data, mask=None, labels=None):
    """
    传统的数据填补方法（不考虑空间结构）
    
    参数:
    - data: 输入数据，形状为 [n_samples, n_features]
    - mask: 掩码，True表示观测值，False表示缺失值
    - labels: 样本标签，用于按类别计算均值
    
    返回:
    - data_filled: 填充后的数据
    - mask: 掩码
    """
    if mask is None:
        mask = ~np.isnan(data)
    
    data_filled = np.copy(data)
    
    if labels is not None and not np.all(np.isnan(labels)):
        # 按类别填充
        print("使用按类别均值填充策略（传统方法）")
        valid_mask = ~np.isnan(labels)
        unique_labels = np.unique(labels[valid_mask])
        
        for label in unique_labels:
            # 获取当前类别的样本索引
            label_indices = np.where((labels == label) & valid_mask)[0]
            if len(label_indices) == 0:
                continue
                
            # 对当前类别的每个特征计算均值
            for i in range(data.shape[1]):
                # 获取当前类别中该特征的有效值
                valid_values = data[label_indices, i][mask[label_indices, i]]
                if len(valid_values) > 0:
                    # 计算当前类别该特征的均值
                    feature_mean = np.mean(valid_values)
                    # 填充当前类别中该特征的缺失值
                    missing_indices = label_indices[~mask[label_indices, i]]
                    if len(missing_indices) > 0:
                        data_filled[missing_indices, i] = feature_mean
    
        # 对于没有标签或标签对应类别样本不足的情况，使用全局均值填充
        still_missing = ~mask & np.isnan(data_filled)
        if np.any(still_missing):
            print("部分缺失值无法通过类别均值填充，使用全局均值填充")
            # 对每个特征计算全局均值
            for i in range(data.shape[1]):
                if np.any(still_missing[:, i]):
                    # 计算该特征的全局均值
                    global_mean = np.nanmean(data[:, i])
                    # 填充仍然缺失的值
                    data_filled[still_missing[:, i], i] = global_mean
    else:
        # 使用全局均值填充
        print("使用全局均值填充策略（传统方法）")
        mean_values = np.nanmean(data, axis=0)
        for i in range(data.shape[1]):
            data_filled[:, i] = np.where(mask[:, i], data[:, i], mean_values[i])
    
    return data_filled, mask


def process_mat_file(file_path, output_dir=None, treat_zeros_as_missing=True, k=5, epochs=200, alpha=0.9, batch_size=1000):
    """
    处理.mat文件，进行子空间感知的缺失值填补
    
    参数:
    - file_path: .mat文件路径
    - output_dir: 输出目录，默认为data/datasets
    - treat_zeros_as_missing: 是否将0视为缺失值
    - k: KNN图中的近邻数量
    - epochs: 训练轮数
    - alpha: 子空间内连接的权重系数 (0-1)
    - batch_size: 批处理大小，用于分批计算相似度矩阵
    
    返回:
    - output_path: 输出文件路径
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件 '{file_path}' 不存在")
        print(f"请确保文件路径正确，可能需要使用绝对路径或正确的相对路径")
        return None
        
    # 加载数据
    print(f"处理文件: {file_path}")
    mat_data = sio.loadmat(file_path)
    
    # 检查fea和gnd是否存在
    if 'fea' not in mat_data or 'gnd' not in mat_data:
        print("错误: 文件中必须包含'fea'和'gnd'字段")
        return None
    
    # 提取数据
    fea = mat_data['fea']
    gnd = mat_data['gnd']
    
    # 处理特征数据(fea)
    print("\n处理特征数据(fea)...")
    # 检查数据维度
    original_shape_fea = fea.shape
    if len(fea.shape) > 2:
        # 如果是图像数据，展平处理
        fea_flat = fea.reshape(original_shape_fea[0], -1)
    else:
        fea_flat = fea
    
    # 创建掩码
    if treat_zeros_as_missing:
        mask_fea = (fea_flat != 0)
    else:
        mask_fea = ~np.isnan(fea_flat)
    
    # 检查缺失值比例
    missing_rate_fea = 1 - np.mean(mask_fea)
    print(f"特征缺失值比例: {missing_rate_fea:.2%}")
    
    # 处理标签数据(gnd)
    print("\n处理标签数据(gnd)...")
    original_shape_gnd = gnd.shape
    
    # 将gnd转换为适合处理的形式
    if len(gnd.shape) == 2 and gnd.shape[1] == 1:
        # 如果gnd是列向量，转换为一维数组
        gnd_flat = gnd.flatten()
    else:
        gnd_flat = gnd
    
    # 创建掩码
    if treat_zeros_as_missing:
        mask_gnd = (gnd_flat != 0)
    else:
        mask_gnd = ~np.isnan(gnd_flat)
    
    # 检查标签中的缺失值
    missing_rate_gnd = 1 - np.mean(mask_gnd)
    print(f"标签缺失值比例: {missing_rate_gnd:.2%}")
    
    # 先填补标签中的缺失值
    if missing_rate_gnd > 0:
        print("\n先填补标签中的缺失值...")
        # 简化标签填补过程，不需要特征辅助
        imputed_gnd_flat = fill_missing_labels(gnd_flat, mask_gnd)
    else:
        imputed_gnd_flat = gnd_flat
    
    # 使用填补后的标签进行子空间感知的GNN特征填补
    print("\n使用子空间感知的GNN进行特征填补...")
    imputed_fea_flat = subspace_aware_imputation(
        fea_flat, imputed_gnd_flat, 
        mask=mask_fea, 
        epochs=epochs, 
        k=k,
        alpha=alpha,
        batch_size=batch_size
    )
    
    # 如果是图像数据，恢复原始形状
    if len(original_shape_fea) > 2:
        imputed_fea = imputed_fea_flat.reshape(original_shape_fea)
    else:
        imputed_fea = imputed_fea_flat
    
    # 恢复标签的原始形状
    if len(original_shape_gnd) == 2 and original_shape_gnd[1] == 1:
        imputed_gnd = imputed_gnd_flat.reshape(original_shape_gnd)
    else:
        imputed_gnd = imputed_gnd_flat
    
    # 设置默认输出目录为data/datasets
    if output_dir is None:
        output_dir = 'data/datasets'
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    file_name = os.path.basename(file_path)
    base_name, ext = os.path.splitext(file_name)
    
    # 构建输出文件名
    if "_subspace_gnn_imputed" not in base_name:
        output_path = os.path.join(output_dir, f"{base_name}_subspace_gnn_imputed{ext}")
    else:
        output_path = os.path.join(output_dir, file_name)
    
    # 保存填补后的数据
    output_data = {
        'fea': imputed_fea,
        'gnd': imputed_gnd
    }
    
    # 复制原始.mat文件中的其他字段
    for key, value in mat_data.items():
        if key not in ['fea', 'gnd'] and not key.startswith('__'):
            output_data[key] = value
    
    sio.savemat(output_path, output_data)
    print(f"填补后的数据已保存至: {output_path}")
    
    return output_path


def fill_missing_labels(labels, mask, features=None):
    """
    填补缺失标签，基于每个类别样本数量相等且相同标签样本连续排列的原理
    
    参数:
    - labels: 标签向量
    - mask: 标签掩码，True表示观测值，False表示缺失值
    - features: 特征矩阵，用于辅助填补（可选，本方法中不使用）
    
    返回:
    - imputed_labels: 填补后的标签
    """
    imputed_labels = np.copy(labels)
    missing_indices = np.where(~mask)[0]
    
    if len(missing_indices) == 0:
        return imputed_labels
    
    valid_indices = np.where(mask)[0]
    
    if len(valid_indices) == 0:
        print("警告: 没有有效标签可用于填补")
        imputed_labels[missing_indices] = 1  # 默认填充为1
        return imputed_labels
    
    # 获取有效标签
    valid_labels = labels[valid_indices]
    unique_labels = np.unique(valid_labels)
    n_classes = len(unique_labels)
    
    # 计算总样本数
    n_samples = len(labels)
    
    # 检查是否符合每个类别样本数量相等的条件
    if n_samples % n_classes == 0:
        samples_per_class = n_samples // n_classes
        print(f"检测到每个类别的样本数量应为: {samples_per_class}")
        
        # 基于样本位置填补标签
        for idx in missing_indices:
            # 确定样本所属类别（基于其在数据集中的位置）
            class_idx = idx // samples_per_class
            if class_idx < n_classes:
                # 使用类别索引获取对应的标签值
                imputed_labels[idx] = unique_labels[class_idx]
            else:
                # 如果计算出的类别索引超出范围，使用最后一个类别
                imputed_labels[idx] = unique_labels[-1]
    else:
        print("警告: 样本总数不能被类别数整除，无法使用基于位置的标签填补")
        print("使用基于最近有效样本的标签填补方法")
        
        # 对于每个缺失标签，查找最近的有效标签
        for idx in missing_indices:
            # 计算与所有有效索引的距离
            distances = np.abs(idx - valid_indices)
            nearest_idx = valid_indices[np.argmin(distances)]
            imputed_labels[idx] = labels[nearest_idx]
    
    # 验证填补结果
    filled_labels = imputed_labels[missing_indices]
    unique_filled = np.unique(filled_labels)
    print(f"填补的标签值: {unique_filled}")
    
    # 检查每个类别的样本数量是否平衡
    if n_samples % n_classes == 0:
        for label in unique_labels:
            count = np.sum(imputed_labels == label)
            expected = samples_per_class
            if count != expected:
                print(f"警告: 标签 {label} 的样本数量为 {count}，期望值为 {expected}")
    
    return imputed_labels


def main():
    parser = argparse.ArgumentParser(description='使用子空间感知的图神经网络进行缺失值填补')
    parser.add_argument('--input', type=str, required=True, help='输入.mat文件路径')
    parser.add_argument('--output_dir', type=str, default='data/datasets', help='输出目录，默认为data/datasets')
    parser.add_argument('--treat_zeros', action='store_true', help='将0值视为缺失值')
    parser.add_argument('--k', type=int, default=10, help='KNN图中的近邻数量')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--alpha', type=float, default=0.9, help='子空间内连接的权重系数 (0-1)')
    parser.add_argument('--batch_size', type=int, default=1000, help='批处理大小，用于分批计算相似度矩阵')
    parser.add_argument('--no_cuda', action='store_true', help='禁用CUDA加速')
    
    args = parser.parse_args()
    
    # 处理CUDA选项
    if args.no_cuda:
        global device
        device = torch.device('cpu')
        print("已禁用CUDA，使用CPU")
    
    # 打印使用示例
    if not os.path.exists(args.input):
        print(f"错误: 文件 '{args.input}' 不存在")
        print("\n使用示例:")
        print("  python subspace_aware_gnn.py --input data/datasets/COIL100_random_zero.mat --treat_zeros")
        print("  python subspace_aware_gnn.py --input data/datasets/ORL_32x32_random_zero.mat --k 15 --epochs 300 --alpha 0.9")
        print("\n可用的数据集文件:")
        for root, dirs, files in os.walk("data/datasets"):
            for file in files:
                if file.endswith(".mat"):
                    print(f"  {os.path.join(root, file)}")
        return
    
    process_mat_file(
        args.input, 
        args.output_dir, 
        treat_zeros_as_missing=args.treat_zeros,
        k=args.k,
        epochs=args.epochs,
        alpha=args.alpha,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main() 