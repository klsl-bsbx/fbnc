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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    print(f"Current CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"Starting execution of {func.__name__}...")
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} finished, elapsed time: {end_time - start_time:.2f} seconds")
        return result
    return wrapper

class SubspaceAwareGNNLayer(nn.Module):
    """
    Subspace-aware Graph Neural Network (GNN) layer.
    This custom layer can learn different transformation weights for different subspaces
    (e.g., different classes) within the data.
    """
    def __init__(self, in_features, out_features, n_subspaces=None):
        super(SubspaceAwareGNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_subspaces = n_subspaces if n_subspaces is not None else 1
        
        # Create a weight matrix for each subspace
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
        
        if self.n_subspaces > 1 and subspace_labels is not None:
            # Subspace-aware mode
            output = torch.zeros(input_features.shape[0], self.out_features, device=input_features.device)
            
            # Process each subspace separately
            for s in range(self.n_subspaces):
                # Get the sample mask for the current subspace
                mask = (subspace_labels == s)
                if not torch.any(mask):
                    continue
                    
                # Get the samples for the current subspace
                subspace_features = input_features[mask]
                
                # Apply the weights for the current subspace
                support = torch.mm(subspace_features, self.weights[s])
                
                # Modify the adjacency matrix to only keep connections within the subspace
                subspace_adj = adj[mask][:, mask]
                
                # Graph convolution operation
                subspace_output = torch.spmm(subspace_adj, support)
                
                # Put the results back into the original output
                output[mask] = subspace_output
                
            return output
        else:
            # Standard GNN mode
            if self.n_subspaces > 1:
                
                support = torch.mm(input_features, self.weights[0])
            else:
                support = torch.mm(input_features, self.weight)
                
            output = torch.spmm(adj, support)
            return output


class SubspaceAwareGNN(nn.Module):
    """
    Subspace-aware Graph Neural Network model
    """
    def __init__(self, input_dim, hidden_dims=[1024, 512, 256], output_dim=None, n_subspaces=None):
        super(SubspaceAwareGNN, self).__init__()
        self.input_dim = input_dim
        self.n_subspaces = n_subspaces if n_subspaces is not None else 1
        
        if output_dim is None:
            output_dim = input_dim
        
        # Build GNN layers
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(SubspaceAwareGNNLayer(prev_dim, hidden_dim, self.n_subspaces))
            prev_dim = hidden_dim
        
        # Output layer
        self.output_layer = SubspaceAwareGNNLayer(prev_dim, output_dim, self.n_subspaces)
        
        # Prediction layer - create a predictor for each subspace
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
        
        # Encode
        h = x
        for layer in self.layers:
            h = F.relu(layer(h, adj, subspace_labels))
            h = F.dropout(h, 0.2, training=self.training)
        
       
        h = self.output_layer(h, adj, subspace_labels)
        
        # Predict original features
        if self.n_subspaces > 1 and subspace_labels is not None:
            # Use a different predictor for each subspace
            output = torch.zeros_like(x)
            for s in range(self.n_subspaces):
                mask = (subspace_labels == s)
                if torch.any(mask):
                    output[mask] = self.predictors[s](h[mask])
            return output
        else:
           
            if self.n_subspaces > 1:
                return self.predictors[0](h)
            else:
                return self.predictor(h)


def ensure_no_missing(data_filled):
    """Ensure data has no missing values"""
    still_missing = np.isnan(data_filled)
    if np.any(still_missing):
        print(f"Still {np.sum(still_missing)} missing values after imputation, applying final fill.")
        
        # Copy data to avoid modifying the original
        result = np.copy(data_filled)
        
        # Process each column separately
        for i in range(result.shape[1]):
            col_mask = np.isnan(result[:, i])
            if np.any(col_mask):
                # Try using column mean
                col_mean = np.nanmean(result[:, i])
                if np.isnan(col_mean):  # Entire column is NaN
                    col_mean = 0
                result[col_mask, i] = col_mean
        
        return result
    return data_filled

def build_subspace_aware_graph(data, labels, k=5, metric='cosine', alpha=0.9, similarity_threshold=0.9, batch_size=1000):
    """
    Builds a subspace-aware graph, weighting connections between same-label samples based on feature similarity.
    """
    from sklearn.neighbors import kneighbors_graph
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
    
    # Handle missing values for distance calculation
    mask = ~np.isnan(data)
    
    # Impute using MICE
    try:
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        import warnings
        warnings.filterwarnings("ignore")
        
        print("Preparing data for graph construction...")
        # Dynamically adjust based on data dimension and subspace characteristics
        if data.shape[1] > 1000:  # High-dimensional data
            n_nearest = min(50, data.shape[1]//20)
        elif data.shape[1] > 100:  # Medium-dimensional
            n_nearest = min(30, data.shape[1]//5)
        else:  # Low-dimensional
            n_nearest = None  # Use all features

        print(f"Using n_nearest_features={n_nearest} for MICE imputation")
        
        # Create MICE imputer
        mice_imputer = IterativeImputer(
            max_iter=5000,
            random_state=42,
            verbose=0,
            n_nearest_features=n_nearest
        )
        
        # Impute using MICE
        print("Imputing data with MICE for graph construction...")
        start_time = time.time()
        
        # Check for valid labels, if any, impute by subspace
        if labels is not None and not np.all(np.isnan(labels)):
            print("Valid labels detected, performing MICE imputation by subspace...")
            data_filled = np.copy(data)
            
            # Get valid labels
            valid_mask = ~np.isnan(labels)
            valid_labels = labels[valid_mask]
            
            # Get unique labels (subspaces)
            unique_labels = np.unique(valid_labels)
            print(f"Detected {len(unique_labels)} subspaces")
            
            # Count samples in each subspace
            subspace_counts = {}
            for label in unique_labels:
                count = np.sum(labels == label)
                subspace_counts[label] = count
                print(f"  Subspace {label}: {count} samples")
            
            # Function to calculate optimal batch size
            def calculate_optimal_batch_size(subspace_size):
                """Calculates the optimal batch size based on subspace size"""
                if subspace_size <= 100:
                   
                    return subspace_size
                elif subspace_size <= 200:
                
                    return subspace_size
                else:
                    # For large subspaces, use a factor of the subspace size or a fixed size
                    for factor in [2, 3, 4, 5]:
                        if subspace_size % factor == 0:
                            batch = subspace_size // factor
                            if 50 <= batch <= 200:
                                return batch
                    
                    # If no suitable factor found, use default
                    return min(200, max(50, subspace_size // 5))
            
            # Perform MICE imputation separately for each subspace
            for label in tqdm(unique_labels, desc="MICE Imputation by Subspace"):
               
                indices = np.where((labels == label) & valid_mask)[0]
                
                if len(indices) == 0:
                    continue
                
                
                subspace_data = data[indices].copy()
                subspace_mask = mask[indices].copy()
                
                # Check for missing values
                if np.all(subspace_mask):
                    print(f"  Subspace {label} has no missing values, skipping MICE imputation.")
                    continue
                
                # Fill with mean first to get initial values
                subspace_mean_filled = np.copy(subspace_data)
                for j in range(subspace_data.shape[1]):
                    col_values = subspace_data[:, j][subspace_mask[:, j]]
                    if len(col_values) > 0:
                        col_mean = np.mean(col_values)
                        subspace_mean_filled[:, j] = np.where(subspace_mask[:, j], 
                                                             subspace_data[:, j], 
                                                             col_mean)
                
                try:
                    # Apply MICE to the current subspace
                    subspace_size = len(indices)
                    
                    # Calculate optimal batch size
                    optimal_batch_size = calculate_optimal_batch_size(subspace_size)
                    
                    # If subspace samples are large, process in batches
                    if subspace_size > optimal_batch_size:
                        print(f"  Subspace {label} has {subspace_size} samples, using batch size {optimal_batch_size}")
                        subspace_filled = np.zeros_like(subspace_data)
                        
                        for j in range(0, subspace_size, optimal_batch_size):
                            end_j = min(j + optimal_batch_size, subspace_size)
                            sub_batch = subspace_mean_filled[j:end_j].copy()
                            sub_batch_mask = subspace_mask[j:end_j]
                            
                            # Only apply MICE to batches with missing values
                            if not np.all(sub_batch_mask):
                                try:
                                    sub_filled = mice_imputer.fit_transform(sub_batch)
                                    
                                    # Check if missing values still remain after imputation
                                    if np.any(np.isnan(sub_filled)):
                                        print(f"  Subspace {label} batch {j//optimal_batch_size + 1} still has missing values after MICE, using mean imputation.")
                                        # Use mean imputation for remaining missing values
                                        for col in range(sub_filled.shape[1]):
                                            col_mask = np.isnan(sub_filled[:, col])
                                            if np.any(col_mask):
                                                col_mean = np.nanmean(sub_filled[:, col])
                                                if np.isnan(col_mean):  
                                                    col_mean = 0
                                                sub_filled[col_mask, col] = col_mean
                                    
                                    subspace_filled[j:end_j] = sub_filled
                                except Exception as e:
                                    print(f"  Subspace {label} batch {j//optimal_batch_size + 1} MICE imputation failed: {e}")
                                    subspace_filled[j:end_j] = sub_batch  
                            else:
                                subspace_filled[j:end_j] = sub_batch
                    else:
                        # Process the entire subspace directly
                        print(f"  Subspace {label} has {subspace_size} samples, processing directly.")
                        subspace_filled = mice_imputer.fit_transform(subspace_mean_filled)
                        
                        # Check if missing values still remain after imputation
                        if np.any(np.isnan(subspace_filled)):
                            print(f"  Subspace {label} still has missing values after MICE, using mean imputation.")
                            # Use mean imputation for remaining missing values
                            for col in range(subspace_filled.shape[1]):
                                col_mask = np.isnan(subspace_filled[:, col])
                                if np.any(col_mask):
                                    col_mean = np.nanmean(subspace_filled[:, col])
                                    if np.isnan(col_mean): 
                                        col_mean = 0
                                    subspace_filled[col_mask, col] = col_mean
                    
                    # Put the imputation results back into the original array
                    data_filled[indices] = subspace_filled
                    
                except Exception as e:
                    print(f"  Subspace {label} MICE imputation failed: {e}, using mean imputation.")
                    data_filled[indices] = subspace_mean_filled
            
        else:
            
            print("No valid labels detected, performing regular batch MICE imputation...")
            
            # Check data size, if too large, process in batches
            n_samples, n_features = data.shape
            if n_samples > 1000:
                print(f"Large data ({n_samples} samples), performing MICE imputation in batches...")
                data_filled = np.zeros_like(data)
                
                
                data_mean_filled = np.copy(data)
                for i in range(data.shape[1]):
                    col_mean = np.nanmean(data[:, i])
                    data_mean_filled[:, i] = np.where(mask[:, i], data[:, i], col_mean)
                
                # Calculate optimal batch size
                batch_size_mice = min(1000, n_samples // 5)  # Default
                for factor in [5, 4, 8, 10]:
                    if n_samples % factor == 0:
                        candidate = n_samples // factor
                        if 100 <= candidate <= 1000:
                            batch_size_mice = candidate
                            break
                
                n_batches = (n_samples + batch_size_mice - 1) // batch_size_mice
                print(f"Using batch size: {batch_size_mice}, total {n_batches} batches")
                
                for i in tqdm(range(0, n_samples, batch_size_mice), desc="MICE Batch Imputation"):
                    end_idx = min(i + batch_size_mice, n_samples)
                    batch_data = data_mean_filled[i:end_idx].copy()  
                    
                    # Mark missing values in the current batch
                    batch_mask = mask[i:end_idx]
                    
                    # Only apply MICE to batches with missing values
                    if not np.all(batch_mask):
                        try:
                            batch_filled = mice_imputer.fit_transform(batch_data)
                            
                            # Check if missing values still remain after imputation
                            if np.any(np.isnan(batch_filled)):
                                print(f"Batch {i//batch_size_mice + 1}/{n_batches} still has missing values after MICE, using mean imputation.")
                                # Use mean imputation for remaining missing values
                                for col in range(batch_filled.shape[1]):
                                    col_mask = np.isnan(batch_filled[:, col])
                                    if np.any(col_mask):
                                        col_mean = np.nanmean(batch_filled[:, col])
                                        if np.isnan(col_mean):  
                                            col_mean = 0
                                        batch_filled[col_mask, col] = col_mean
                                        
                            data_filled[i:end_idx] = batch_filled
                        except Exception as e:
                            print(f"Batch {i//batch_size_mice + 1}/{n_batches} MICE imputation failed: {e}")
                            data_filled[i:end_idx] = batch_data  
                    else:
                        data_filled[i:end_idx] = batch_data
            else:
                # Smaller data, process directly
                data_filled = mice_imputer.fit_transform(np.where(np.isnan(data), np.nanmean(data, axis=0), data))
                
                # Check if missing values still remain after imputation
                if np.any(np.isnan(data_filled)):
                    print("Still has missing values after MICE, using mean imputation.")
                    # Use mean imputation for remaining missing values
                    for col in range(data_filled.shape[1]):
                        col_mask = np.isnan(data_filled[:, col])
                        if np.any(col_mask):
                            col_mean = np.nanmean(data_filled[:, col])
                            if np.isnan(col_mean):  # Entire column is NaN
                                col_mean = 0
                            data_filled[col_mask, col] = col_mean
        
        end_time = time.time()
        print(f"MICE imputation completed, elapsed time: {end_time - start_time:.2f} seconds")
        
    except (ImportError, ValueError) as e:
    
        print(f"MICE method failed, error: {e}, falling back to mean imputation.")
        print("Imputing data with mean for graph construction...")
        data_filled = np.copy(data)
        for i in tqdm(range(data.shape[1]), desc="Mean Imputation"):
            col_mean = np.nanmean(data[:, i])
            data_filled[:, i] = np.where(mask[:, i], data[:, i], col_mean)
    
   
    data_filled = ensure_no_missing(data_filled)
    
    print("Building KNN graph...")
    start_time = time.time()
    
   
    n_samples = data.shape[0]
    if n_samples > 5000 and torch.cuda.is_available():
        print(f"Large number of samples ({n_samples}), using GPU-accelerated KNN...")
        try:
            # Use GPU accelerated KNN calculation
            data_tensor = torch.FloatTensor(data_filled).to(device)
            adj = torch.zeros((n_samples, n_samples), device=device)
            
            # Use subspace-aware batch size
            batch_size_knn = calculate_subspace_aware_batch_size(labels, default_size=360)
            print(f"Using subspace-aware batch size: {batch_size_knn}")
            
            for i in tqdm(range(0, n_samples, batch_size_knn), desc="Calculating KNN (GPU)"):
                end_i = min(i + batch_size_knn, n_samples)
                batch_i = data_tensor[i:end_i]
                
                # Calculate distances between current batch and all samples
                if metric == 'cosine':
                    # Calculate cosine similarity
                    batch_norm = torch.norm(batch_i, dim=1, keepdim=True)
                    all_norm = torch.norm(data_tensor, dim=1, keepdim=True)
                    
                    # Avoid division by zero
                    batch_norm[batch_norm == 0] = 1e-8
                    all_norm[all_norm == 0] = 1e-8
                    
                    batch_normalized = batch_i / batch_norm
                    all_normalized = data_tensor / all_norm
                    
                    # Calculate similarity matrix (larger is more similar)
                    sim_matrix = torch.mm(batch_normalized, all_normalized.t())
                    
                    # Find top k values for each row (excluding self)
                    for j in range(i, end_i):
                        row_idx = j - i
                        sim_row = sim_matrix[row_idx]
                        sim_row[j] = -1  # Exclude self
                        
                        # Find indices of the top k largest values
                        _, topk_indices = torch.topk(sim_row, k)
                        
                        # Set adjacency matrix
                        adj[j, topk_indices] = 1
                else:
                    # Euclidean distance
                    for j in range(i, end_i):
                        row_idx = j - i
                        diff = data_tensor[j:j+1] - data_tensor
                        dist = torch.sum(diff * diff, dim=1)
                        
                        # Exclude self
                        dist[j] = float('inf')
                        
                        # Find indices of the top k smallest values
                        _, topk_indices = torch.topk(dist, k, largest=False)
                        
                        # Set adjacency matrix
                        adj[j, topk_indices] = 1
            
            adj = adj.cpu().numpy()
            
        except Exception as e:
            print(f"GPU accelerated KNN failed: {e}, falling back to CPU calculation.")
            knn = kneighbors_graph(data_filled, k, metric=metric, include_self=False)
            adj = knn.toarray()
    else:
        knn = kneighbors_graph(data_filled, k, metric=metric, include_self=False)
        adj = knn.toarray()
    
    end_time = time.time()
    print(f"KNN graph construction completed, elapsed time: {end_time - start_time:.2f} seconds")
    
    # Get KNN indices for each sample
    n_samples = data.shape[0]
    knn_indices = []
    for i in range(n_samples):
        neighbors = np.nonzero(adj[i])[0]
        knn_indices.append(neighbors)
    
    # If label information is available, enhance intra-subspace connections
    if labels is not None:
        print("Enhancing intra-subspace connections...")
        
        # Calculate feature similarity between all pairs of samples
        n_samples = data.shape[0]
        
        # Find valid labels
        valid_mask = ~np.isnan(labels)
        valid_indices = np.where(valid_mask)[0]
        n_valid = len(valid_indices)
        
        print(f"Calculating similarity matrix for {n_valid} valid samples...")
        
        # Create label mask matrix
        from scipy.sparse import lil_matrix
        label_mask = lil_matrix((n_samples, n_samples), dtype=np.float32)
        
        # Preprocess labels, create label to index mapping
        label_to_indices = {}
        for i, idx in enumerate(valid_indices):
            lbl = labels[idx]
            if lbl not in label_to_indices:
                label_to_indices[lbl] = []
            label_to_indices[lbl].append(idx)
        
        print(f"Processing {len(label_to_indices)} groups of samples with different labels...")
        
        # Process each label group separately
        for label, indices in tqdm(label_to_indices.items(), desc="Processing Label Groups"):
            n_indices = len(indices)
            
            # Calculate optimal batch size
            optimal_batch_size = min(batch_size, max(72, n_indices // 2))
            
            # If too many samples with the same label, process in batches
            if n_indices > optimal_batch_size:
                for i in range(0, n_indices, optimal_batch_size):
                    batch_indices = indices[i:min(i+optimal_batch_size, n_indices)]
                    process_batch(batch_indices, indices, data_filled, label_mask, 
                                 similarity_threshold, metric, device)
            else:
                # Process directly
                process_batch(indices, indices, data_filled, label_mask, 
                             similarity_threshold, metric, device)
        
        print(f"Processing {len(label_to_indices)} groups of samples with different labels...")
        
        # Process each label group separately
        for label, indices in tqdm(label_to_indices.items(), desc="Processing Label Groups"):
            n_indices = len(indices)
            
            # Calculate subspace-aware batch size
            # For samples with the same label, we want the batch size to be a factor of the number of samples to avoid splitting
            # Try to find a factor close to but not exceeding the original batch size
            factors = []
            for i in range(1, int(np.sqrt(n_indices)) + 1):
                if n_indices % i == 0:
                    factors.append(i)
                    if i != n_indices // i:
                        factors.append(n_indices // i)
            factors.sort()
            
            # Select the largest factor closest to but not exceeding batch_size
            optimal_batch_size = 1  
            for factor in factors:
                if factor <= batch_size:
                    optimal_batch_size = factor
                else:
                    break
            
          
            if optimal_batch_size < batch_size * 0.2:
                optimal_batch_size = min(batch_size, n_indices)
            
            print(f"  Label {label} samples: {n_indices}, using batch size: {optimal_batch_size} (Factor: {n_indices % optimal_batch_size == 0})")
            
            # If too many samples with the same label, process in batches
            if n_indices > optimal_batch_size:
                for i in range(0, n_indices, optimal_batch_size):
                    batch_indices = indices[i:min(i+optimal_batch_size, n_indices)]
                    process_batch(batch_indices, indices, data_filled, label_mask, 
                                 similarity_threshold, metric, device)
            else:
                process_batch(indices, indices, data_filled, label_mask, 
                             similarity_threshold, metric, device)
        
        
        
        
        print("Merging KNN graph and subspace connections...")
        
        label_mask = label_mask.toarray()
        
        # Enhance intra-subspace connections using weighted average
        adj = alpha * adj + (1 - alpha) * label_mask
    
    # Ensure symmetry
    adj = np.maximum(adj, adj.T)
    
    print("Converting to PyTorch tensor and normalizing...")
   
    adj_tensor = torch.FloatTensor(adj)
    
    # Add self-loops
    adj_tensor = adj_tensor + torch.eye(adj_tensor.shape[0])
    
    # Degree matrix
    D = torch.diag(torch.sum(adj_tensor, dim=1))
    
    # Normalized adjacency matrix: D^(-1/2) * A * D^(-1/2)
    D_inv_sqrt = torch.pow(D, -0.5)
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0
    adj_normalized = torch.mm(torch.mm(D_inv_sqrt, adj_tensor), D_inv_sqrt)
    
    return adj_normalized, knn_indices

def process_batch(batch_indices, all_indices, data_filled, label_mask, similarity_threshold, metric, device):
    """
    Processes the similarity calculation between a batch of samples and all samples with the same label.
    """
    batch_data = data_filled[batch_indices]
    all_data = data_filled[all_indices]
    
    if torch.cuda.is_available():
        batch_tensor = torch.FloatTensor(batch_data).to(device)
        all_tensor = torch.FloatTensor(all_data).to(device)
        
        if metric == 'cosine':
            # Calculate cosine similarity
            batch_norm = torch.norm(batch_tensor, dim=1, keepdim=True)
            all_norm = torch.norm(all_tensor, dim=1, keepdim=True)
            
            # Avoid division by zero
            batch_norm[batch_norm == 0] = 1e-8
            all_norm[all_norm == 0] = 1e-8
            
            batch_normalized = batch_tensor / batch_norm
            all_normalized = all_tensor / all_norm
            
            # Calculate similarity matrix
            batch_sim = torch.mm(batch_normalized, all_normalized.t()).cpu().numpy()
            
            # Normalize similarity to [0, 1]
            batch_sim = (batch_sim + 1) / 2
        else:
            # Euclidean distance
            batch_squared = torch.sum(batch_tensor ** 2, dim=1, keepdim=True)
            all_squared = torch.sum(all_tensor ** 2, dim=1, keepdim=True)
            
            cross_term = torch.mm(batch_tensor, all_tensor.t())
            dist_matrix = batch_squared + all_squared.t() - 2 * cross_term
            dist_matrix = torch.clamp(dist_matrix, min=0)  # Avoid negative values due to numerical errors
            dist_matrix = torch.sqrt(dist_matrix).cpu().numpy()
            
            # Convert distance to similarity
            max_dist = np.max(dist_matrix) if np.max(dist_matrix) > 0 else 1.0
            batch_sim = 1 - (dist_matrix / max_dist)
    else:
       
        if metric == 'cosine':
            batch_sim = cosine_similarity(batch_data, all_data)
            batch_sim = (batch_sim + 1) / 2
        else:
            dist_matrix = euclidean_distances(batch_data, all_data)
            max_dist = np.max(dist_matrix) if np.max(dist_matrix) > 0 else 1.0
            batch_sim = 1 - (dist_matrix / max_dist)
    
    # Build label mask matrix based on feature similarity
    for idx, orig_idx in enumerate(batch_indices):
        for j_idx, j in enumerate(all_indices):
            if orig_idx != j:  # Do not connect to self
                # Connect only if feature similarity is above threshold
                sim = batch_sim[idx, j_idx]
                if sim >= similarity_threshold:
                    # Use similarity as connection weight
                    label_mask[orig_idx, j] = sim

def masked_mse_loss(output, target, mask):
    """
    Calculates Mean Squared Error (MSE) loss with a mask.
    """
    diff = (output - target) * mask
    return torch.sum(diff ** 2) / torch.sum(mask)


def enhanced_masked_loss(output, target, mask, subspace_labels=None, feature_weights=None, 
                         use_ssim=True, img_size=32, alpha_mse=0.7, alpha_ssim=0.3, 
                         alpha_subspace=1):
    """
    Enhanced loss function, combining MSE, structural similarity, and weighting mechanisms.
    """
    n_samples = output.shape[0]
    n_features = output.shape[1]
    device = output.device
    
    # Base MSE loss
    diff = (output - target) * mask
    
    # Apply subspace weights
    if subspace_labels is not None:
        # Calculate average error for each subspace
        unique_subspaces = torch.unique(subspace_labels)
        n_subspaces = len(unique_subspaces)
        
        # Create subspace weight matrix
        subspace_weights = torch.ones(n_samples, device=device)
        
        # Calculate average loss for each subspace
        subspace_losses = []
        for s in unique_subspaces:
            s_mask = (subspace_labels == s)
            if not torch.any(s_mask):
                continue
                
           
            s_diff = diff[s_mask]
            s_mask_vals = mask[s_mask]
            if torch.sum(s_mask_vals) > 0:
                s_loss = torch.sum(s_diff ** 2) / torch.sum(s_mask_vals)
                subspace_losses.append((s, s_loss))
        
        # Allocate weights based on loss magnitude, focusing more on difficult-to-impute subspaces
        if len(subspace_losses) > 1:
            # Normalize subspace losses
            s_losses = torch.tensor([l for _, l in subspace_losses], device=device)
            min_loss = torch.min(s_losses)
            max_loss = torch.max(s_losses)
            
            
            if min_loss != max_loss:
                norm_losses = (s_losses - min_loss) / (max_loss - min_loss)
                
                # Convert normalized losses to weights
                s_weights = 1.0 + norm_losses  
                
                # Apply weights to corresponding subspaces
                for i, (s, _) in enumerate(subspace_losses):
                    subspace_weights[subspace_labels == s] = s_weights[i]
        
       
        diff = diff * subspace_weights.view(-1, 1)
    
   
    # Calculate missing rate of features
    feature_missing_rate = 1 - torch.mean(mask, dim=0)
    
    # Calculate final MSE loss
    mse_loss = torch.sum(diff ** 2) / torch.sum(mask)
    
    # Structural Similarity Loss (SSIM)
    if use_ssim and n_features == img_size * img_size:  # Only use SSIM for image data
        try:
            ssim_loss = 0
            valid_samples = 0
            
            # Reshape to image format for SSIM calculation
            output_images = output.view(-1, 1, img_size, img_size)  # [B, C, H, W]
            target_images = target.view(-1, 1, img_size, img_size)
            mask_images = mask.view(-1, 1, img_size, img_size)
            
            # Calculate SSIM per sample
            for i in range(n_samples):
                # Only calculate SSIM for samples with mask coverage > 50%
                if torch.mean(mask_images[i]) >= 0.5:
                    sample_ssim = 1 - compute_ssim(output_images[i], target_images[i], mask_images[i])
                    ssim_loss += sample_ssim
                    valid_samples += 1
            
            # Average SSIM loss
            if valid_samples > 0:
                ssim_loss = ssim_loss / valid_samples
            else:
                ssim_loss = torch.tensor(0.0, device=device)
                
            # Combine MSE and SSIM loss
            combined_loss = alpha_mse * mse_loss + alpha_ssim * ssim_loss
            return combined_loss
            
        except Exception as e:
            print(f"SSIM calculation error: {e}, falling back to MSE loss.")
            return mse_loss
    
    return mse_loss

def compute_ssim(img1, img2, mask=None, window_size=11, sigma=1.5):
    """
    Calculates the masked Structural Similarity Index (SSIM).
    """
  
    device = img1.device
    
    # Create Gaussian window
    window = create_gaussian_window(window_size, sigma).to(device)
    
    # Apply mask
    if mask is not None:
        img1 = img1 * mask
        img2 = img2 * mask
    
    # Calculate means
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=1)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=1)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # Calculate variances and covariance
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=1) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=1) - mu1_mu2
    
    # SSIM constants
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Calculate SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    # If mask is present, consider only masked regions
    if mask is not None:
        # Downsample mask to match SSIM map
        mask_downsampled = F.conv2d(mask, window, padding=window_size//2, groups=1)
        mask_downsampled = (mask_downsampled > 0.5).float()  # Binarize
        
        # Calculate average SSIM in the masked area
        ssim_masked = (ssim_map * mask_downsampled).sum() / (mask_downsampled.sum() + 1e-8)
        return ssim_masked
    
    return ssim_map.mean()


def create_gaussian_window(window_size, sigma):
    """
    Creates a Gaussian window for SSIM calculation.
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
    Calculates a subspace-aware batch size, ensuring the batch size is an integer multiple of the number of samples in a subspace.
    """
   
    valid_mask = ~np.isnan(labels)
    if not np.any(valid_mask):
        return default_size 
    
    # Get unique labels
    unique_labels = np.unique(labels[valid_mask])
    
    # Calculate the number of samples in each subspace
    subspace_sizes = []
    for label in unique_labels:
        size = np.sum(labels == label)
        subspace_sizes.append(size)
    
    
    if len(subspace_sizes) == 1:
        subspace_size = subspace_sizes[0]
        # Choose an appropriate multiplier
        for multiplier in range(min_multiplier, max_multiplier + 1):
            if subspace_size * multiplier <= default_size * 1.5:
                return subspace_size * multiplier
        return subspace_size
    
    # Find the largest subspace size
    max_subspace_size = max(subspace_sizes)
    
    # Find the smallest subspace size
    min_subspace_size = min(subspace_sizes)
    
    # Try to find a batch size that is divisible by all subspace sizes
    # First, try using multiples of the greatest common divisor (GCD)
    
    def find_gcd(numbers):
        return reduce(gcd, numbers)
    
    common_divisor = find_gcd(subspace_sizes)
    
    if common_divisor >= 10:
        # Choose a multiple of the common divisor that is close to but does not exceed the default size
        multiplier = default_size // common_divisor
        if multiplier < min_multiplier:
            multiplier = min_multiplier
        return common_divisor * multiplier
    
    # If no suitable common divisor, try to find a batch size divisible by the largest subspace size
    for multiplier in range(min_multiplier, max_multiplier + 1):
        batch_size = max_subspace_size * multiplier
        if batch_size >= default_size * 0.7 and batch_size <= default_size * 1.5:
            return batch_size

    return ((max_subspace_size + default_size - 1) // default_size) * default_size



@timer
def subspace_aware_imputation(data, labels, mask=None, n_subspaces=None, epochs=200, lr=0.001, k=5, hidden_dims=[1024, 512, 256], alpha=0.9, verbose=True, batch_size=1000, use_enhanced_loss=True, img_size=32):
    """
    Performs missing value imputation using a subspace-aware Graph Neural Network.
    """
    if mask is None:
        mask = ~np.isnan(data)
    
  
    print("Preparing data...")
    data_filled, _ = prepare_data_for_imputation(data, mask, labels)
    
    # Process labels
    print("Processing labels...")
    valid_labels = ~np.isnan(labels)
    if np.any(valid_labels):
        # Convert labels to integer categories
        unique_labels = np.unique(labels[valid_labels])
        label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
        
        # Create subspace label array
        subspace_labels = np.zeros(labels.shape, dtype=int)
        for i, lbl in enumerate(labels):
            if not np.isnan(lbl):
                subspace_labels[i] = label_map[lbl]
            else:
                
                subspace_labels[i] = 0
        
        # Determine number of subspaces
        if n_subspaces is None:
            n_subspaces = len(unique_labels)
    else:
      
        subspace_labels = np.zeros(labels.shape, dtype=int)
        if n_subspaces is None:
            n_subspaces = 1
    
    print(f"Detected {n_subspaces} subspaces")
    
    # Build subspace-aware graph
    adj, knn_indices = build_subspace_aware_graph(data_filled, labels, k=k, alpha=alpha, batch_size=batch_size)
    
    print("Normalizing data...")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_filled)
    
  
    print("Converting data to PyTorch tensors...")
    data_tensor = torch.FloatTensor(data_scaled).to(device)
    mask_tensor = torch.FloatTensor(mask.astype(float)).to(device)
    adj_tensor = adj.to(device)
    subspace_tensor = torch.LongTensor(subspace_labels).to(device)
    
    
    feature_weights = None
    
    # Create model
    print("Creating GNN model...")
    model = SubspaceAwareGNN(
        input_dim=data.shape[1], 
        hidden_dims=hidden_dims, 
        n_subspaces=n_subspaces
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Train model
    model.train()
    pbar = tqdm(range(epochs)) if verbose else range(epochs)
    losses = []
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=verbose)
    
    print("Starting GNN model training...")
    train_start_time = time.time()
    
    for epoch in pbar:
        # Forward pass
        optimizer.zero_grad()
        output = model(data_tensor, adj_tensor, subspace_tensor)
        
        # Calculate loss
        if use_enhanced_loss:
            loss = enhanced_masked_loss(
                output=output, 
                target=data_tensor, 
                mask=mask_tensor,
                subspace_labels=subspace_tensor,
                feature_weights=None,  # Feature weights no longer used
                use_ssim=(data.shape[1] == img_size * img_size),  # Use SSIM only for images
                img_size=img_size
            )
        else:
            loss = masked_mse_loss(output, data_tensor, mask_tensor)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update learning rate
        scheduler.step(loss)
        
        # Update progress bar
        losses.append(loss.item())
        if verbose:
            pbar.set_description(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
        
        # Update imputed values
        with torch.no_grad():
            model.eval()
            output = model(data_tensor, adj_tensor, subspace_tensor)
            # Only update missing values
            data_tensor = data_tensor * mask_tensor + output * (1 - mask_tensor)
            model.train()
    
    train_end_time = time.time()
    print(f"GNN model training completed, elapsed time: {train_end_time - train_start_time:.2f} seconds")
    
    # Final prediction
    print("Generating final predictions...")
    model.eval()
    with torch.no_grad():
        output = model(data_tensor, adj_tensor, subspace_tensor)
        # Only update missing values
        final_output = data_tensor * mask_tensor + output * (1 - mask_tensor)
    
    # Inverse scaling
    print("Inverse transforming data...")
    imputed_data_scaled = final_output.cpu().numpy()
    imputed_data = scaler.inverse_transform(imputed_data_scaled)
    
    # Only replace missing values
    result = np.copy(data)
    result[~mask] = imputed_data[~mask]
    
    # Visualize loss curve
    if verbose:
        print("Generating loss curve...")
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
    Prepares data for imputation, considering the spatial structure of images, using the MICE method for imputation.
    """
    if mask is None:
        mask = ~np.isnan(data)
    
    data_filled = np.copy(data)
    n_samples = data.shape[0]
    n_features = data.shape[1]
    
    # Check if it's a square image
    if img_size * img_size != n_features:
        print(f"Warning: Number of features {n_features} does not equal {img_size}x{img_size}={img_size*img_size}, using traditional imputation method.")
        return prepare_data_for_imputation_traditional(data, mask, labels)
    
    # Reshape data to image shape for spatial imputation
    data_reshaped = data.reshape(n_samples, img_size, img_size)
    mask_reshaped = mask.reshape(n_samples, img_size, img_size)
    data_filled_reshaped = np.copy(data_reshaped)
    
    try:
    
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        from sklearn.linear_model import BayesianRidge
        
        print("Using MICE method for image data imputation.")
        
        # Process each sample individually
        print(f"Starting MICE imputation for {n_samples} samples...")
        start_time = time.time()
        
        for idx in tqdm(range(n_samples), desc="MICE Imputation Progress"):
            # Get current image
            img = data_reshaped[idx]
            img_mask = mask_reshaped[idx]
            
            # If image has no missing values, skip
            if np.all(img_mask):
                continue
                
            
            
            # Reshape image to a 2D matrix, each row is a pixel, each column is a feature
            img_2d = img.reshape(img_size*img_size, 1)
            img_mask_2d = img_mask.reshape(img_size*img_size, 1)
            
            # Use BayesianRidge as base estimator
            estimator = BayesianRidge()
            imputer = IterativeImputer(
                estimator=estimator,
                max_iter=5000,
                random_state=42,
                verbose=0,
                n_nearest_features=min(300, img_size),  # Use 10 nearest features or all features
                skip_complete=True  # Skip columns with no missing values
            )
            
            # To add feature dimension, we add pixel position information
            rows, cols = np.mgrid[0:img_size, 0:img_size]
            positions = np.column_stack([rows.ravel(), cols.ravel()])
            
            # Combine position information and pixel values
            features = np.column_stack([positions, img_2d])
            
            # Create missing value mask
            missing_mask = ~img_mask_2d.ravel()
            
            # If missing value ratio is too high, MICE might not work well, perform simple fill first
            if np.mean(missing_mask) > 0.5:
                # Use column mean for initial filling
                for j in range(img_size):
                    valid_values = img[:, j][img_mask[:, j]]
                    if len(valid_values) > 0:
                        column_mean = np.mean(valid_values)
                        missing_rows = np.where(~img_mask[:, j])[0]
                        for i in missing_rows:
                            features[i*img_size+j, 2] = column_mean
            
            # Apply MICE
            imputed_features = imputer.fit_transform(features)
            
            # Extract imputed pixel values and reshape back to image
            imputed_img = imputed_features[:, 2].reshape(img_size, img_size)
            
            # Only replace missing values
            data_filled_reshaped[idx][~img_mask] = imputed_img[~img_mask]
            
        end_time = time.time()
        print(f"MICE imputation completed, elapsed time: {end_time - start_time:.2f} seconds")
        
    except (ImportError, ValueError) as e:
        print(f"MICE method failed, error: {e}, falling back to traditional imputation method.")
        
        # Fallback to traditional imputation strategy
        for idx in tqdm(range(n_samples), desc="Traditional Imputation Progress"):
            # For each column in the image
            for j in range(img_size):
                # Get valid values for the current column
                valid_values = data_reshaped[idx, :, j][mask_reshaped[idx, :, j]]
                
                # Calculate column mean and impute missing values in that column
                if len(valid_values) > 0:
                    column_mean = np.mean(valid_values)
                    # Find missing positions in this column
                    missing_rows = np.where(~mask_reshaped[idx, :, j])[0]
                    # Impute missing values
                    for i in missing_rows:
                        data_filled_reshaped[idx, i, j] = column_mean
        
        # For remaining missing values, try using row mean imputation
        still_missing = ~mask_reshaped & np.isnan(data_filled_reshaped)
        if np.any(still_missing):
            print("Some missing values could not be imputed by column mean, attempting row mean imputation.")
            for idx in tqdm(range(n_samples), desc="Row Mean Imputation"):
                # For each row in the image
                for i in range(img_size):
                    # Get valid values for the current row
                    valid_values = data_reshaped[idx, i, :][mask_reshaped[idx, i, :]]
                    
                    # If valid values exist, calculate row mean and impute missing values in that row
                    if len(valid_values) > 0:
                        row_mean = np.mean(valid_values)
                       
                        missing_cols = np.where(~mask_reshaped[idx, i, :] & np.isnan(data_filled_reshaped[idx, i, :]))[0]
                      
                        for j in missing_cols:
                            data_filled_reshaped[idx, i, j] = row_mean
        
        # For remaining missing values, use overall image mean imputation
        still_missing = ~mask_reshaped & np.isnan(data_filled_reshaped)
        if np.any(still_missing):
            print("Some missing values could not be imputed by row/column mean, using overall image mean imputation.")
            for idx in tqdm(range(n_samples), desc="Overall Mean Imputation"):
                # Get valid values for the current image
                valid_values = data_reshaped[idx][mask_reshaped[idx]]
                
                # If valid values exist, calculate overall mean and impute remaining missing values
                if len(valid_values) > 0:
                    image_mean = np.mean(valid_values)
                    # Find remaining missing positions in this image
                    missing_pixels = still_missing[idx]
                    # Impute missing values
                    data_filled_reshaped[idx][missing_pixels] = image_mean
                else:
                    # If the entire image has no valid values, use the mean of all images
                    all_valid_values = data_reshaped[mask_reshaped]
                    if len(all_valid_values) > 0:
                        global_mean = np.mean(all_valid_values)
                        data_filled_reshaped[idx][missing_pixels] = global_mean
    
    # Check if there are still missing values
    still_missing = np.isnan(data_filled_reshaped)
    if np.any(still_missing):
        print("Still missing values, using global mean fill.")
        global_mean = np.nanmean(data_reshaped)
        data_filled_reshaped[still_missing] = global_mean
    
    # Reshape the imputed data back to original flat format
    data_filled = data_filled_reshaped.reshape(n_samples, n_features)
    print("Data imputation completed, preparing to build graph...")
    
    return data_filled, mask


def prepare_data_for_imputation_traditional(data, mask=None, labels=None):
    """
    Traditional data imputation method (without considering spatial structure).
    """
    if mask is None:
        mask = ~np.isnan(data)
    
    data_filled = np.copy(data)
    
    if labels is not None and not np.all(np.isnan(labels)):
        # Impute by category
        print("Using category-wise mean imputation strategy (traditional method).")
        valid_mask = ~np.isnan(labels)
        unique_labels = np.unique(labels[valid_mask])
        
        for label in unique_labels:
            # Get sample indices for the current category
            label_indices = np.where((labels == label) & valid_mask)[0]
            if len(label_indices) == 0:
                continue
                
            # Calculate mean for each feature in the current category
            for i in range(data.shape[1]):
                # Get valid values for this feature in the current category
                valid_values = data[label_indices, i][mask[label_indices, i]]
                if len(valid_values) > 0:
                    # Calculate mean of this feature for the current category
                    feature_mean = np.mean(valid_values)
                    # Impute missing values for this feature in the current category
                    missing_indices = label_indices[~mask[label_indices, i]]
                    if len(missing_indices) > 0:
                        data_filled[missing_indices, i] = feature_mean
    
     
        still_missing = ~mask & np.isnan(data_filled)
        if np.any(still_missing):
            print("Some missing values could not be imputed by category mean, using global mean imputation.")
            # Calculate global mean for each feature
            for i in range(data.shape[1]):
                if np.any(still_missing[:, i]):
                    # Calculate global mean for this feature
                    global_mean = np.nanmean(data[:, i])
                    # Impute remaining missing values
                    data_filled[still_missing[:, i], i] = global_mean
    else:
        # Use global mean imputation
        print("Using global mean imputation strategy (traditional method).")
        mean_values = np.nanmean(data, axis=0)
        for i in range(data.shape[1]):
            data_filled[:, i] = np.where(mask[:, i], data[:, i], mean_values[i])
    
    return data_filled, mask

def process_mat_file(file_path, output_dir=None, treat_zeros_as_missing=True, k=5, epochs=200, alpha=0.9, batch_size=1000):
    """
    Processes .mat files, performing subspace-aware missing value imputation.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        print(f"Please ensure the file path is correct, you might need to use an absolute path or correct relative path.")
        return None
        
    # Load data
    print(f"Processing file: {file_path}")
    mat_data = sio.loadmat(file_path)
    
    # Check if 'fea' and 'gnd' exist
    if 'fea' not in mat_data or 'gnd' not in mat_data:
        print("Error: The file must contain 'fea' and 'gnd' fields.")
        return None
    
    # Extract data
    fea = mat_data['fea']
    gnd = mat_data['gnd']
    
    # Process feature data (fea)
    print("\nProcessing feature data (fea)...")
    # Check data dimension
    original_shape_fea = fea.shape
    if len(fea.shape) > 2:
        # If it's image data, flatten it
        fea_flat = fea.reshape(original_shape_fea[0], -1)
    else:
        fea_flat = fea
    
    # Create mask
    if treat_zeros_as_missing:
        mask_fea = (fea_flat != 0)
    else:
        mask_fea = ~np.isnan(fea_flat)
    
    # Check missing value rate
    missing_rate_fea = 1 - np.mean(mask_fea)
    print(f"Feature missing value rate: {missing_rate_fea:.2%}")
    
    # Process label data (gnd)
    print("\nProcessing label data (gnd)...")
    original_shape_gnd = gnd.shape
    
    # Convert gnd to a suitable format for processing
    if len(gnd.shape) == 2 and gnd.shape[1] == 1:
        # If gnd is a column vector, convert to a 1D array
        gnd_flat = gnd.flatten()
    else:
        gnd_flat = gnd
    
    # Create mask
    if treat_zeros_as_missing:
        mask_gnd = (gnd_flat != 0)
    else:
        mask_gnd = ~np.isnan(gnd_flat)
    
    # Check for missing values in labels
    missing_rate_gnd = 1 - np.mean(mask_gnd)
    print(f"Label missing value rate: {missing_rate_gnd:.2%}")
    
    # Impute missing values in labels first
    if missing_rate_gnd > 0:
        print("\nImputing missing values in labels first...")
        # Simplify label imputation process, no need for feature assistance
        imputed_gnd_flat = fill_missing_labels(gnd_flat, mask_gnd)
    else:
        imputed_gnd_flat = gnd_flat
    
    # Perform subspace-aware GNN feature imputation using the imputed labels
    print("\nPerforming subspace-aware GNN for feature imputation...")
    imputed_fea_flat = subspace_aware_imputation(
        fea_flat, imputed_gnd_flat, 
        mask=mask_fea, 
        epochs=epochs, 
        k=k,
        alpha=alpha,
        batch_size=batch_size
    )
    
    # If it was image data, restore original shape
    if len(original_shape_fea) > 2:
        imputed_fea = imputed_fea_flat.reshape(original_shape_fea)
    else:
        imputed_fea = imputed_fea_flat
    
    # Restore label's original shape
    if len(original_shape_gnd) == 2 and original_shape_gnd[1] == 1:
        imputed_gnd = imputed_gnd_flat.reshape(original_shape_gnd)
    else:
        imputed_gnd = imputed_gnd_flat
    
    
    if output_dir is None:
        output_dir = 'data/datasets'
    
   
    os.makedirs(output_dir, exist_ok=True)
    
    file_name = os.path.basename(file_path)
    base_name, ext = os.path.splitext(file_name)
    
    # Construct output filename
    if "_subspace_gnn_imputed" not in base_name:
        output_path = os.path.join(output_dir, f"{base_name}_subspace_gnn_imputed{ext}")
    else:
        output_path = os.path.join(output_dir, file_name)
    
    # Save imputed data
    output_data = {
        'fea': imputed_fea,
        'gnd': imputed_gnd
    }
    
  
    for key, value in mat_data.items():
        if key not in ['fea', 'gnd'] and not key.startswith('__'):
            output_data[key] = value
    
    sio.savemat(output_path, output_data)
    print(f"Imputed data saved to: {output_path}")
    
    return output_path


def fill_missing_labels(labels, mask, features=None):
    """
    Imputes missing labels, based on the principle that each category has an equal number of samples and same-label samples are arranged contiguously.
    """
    imputed_labels = np.copy(labels)
    missing_indices = np.where(~mask)[0]
    
    if len(missing_indices) == 0:
        return imputed_labels
    
    valid_indices = np.where(mask)[0]
    
    if len(valid_indices) == 0:
        print("Warning: No valid labels available for imputation.")
        imputed_labels[missing_indices] = 1  # Default fill with 1
        return imputed_labels
    
    # Get valid labels
    valid_labels = labels[valid_indices]
    unique_labels = np.unique(valid_labels)
    n_classes = len(unique_labels)
    
    # Calculate total number of samples
    n_samples = len(labels)
    
    # Check if the condition of equal number of samples per category is met
    if n_samples % n_classes == 0:
        samples_per_class = n_samples // n_classes
        print(f"Detected expected samples per class: {samples_per_class}")
        
        # Impute labels based on sample position
        for idx in missing_indices:
           
            class_idx = idx // samples_per_class
            if class_idx < n_classes:
             
                imputed_labels[idx] = unique_labels[class_idx]
            else:
               
                imputed_labels[idx] = unique_labels[-1]
    else:
        print("Warning: Total number of samples is not divisible by the number of classes, cannot use position-based label imputation.")
        print("Using label imputation based on the nearest valid sample.")
        
 
        for idx in missing_indices:
          
            distances = np.abs(idx - valid_indices)
            nearest_idx = valid_indices[np.argmin(distances)]
            imputed_labels[idx] = labels[nearest_idx]
    
    # Verify imputation results
    filled_labels = imputed_labels[missing_indices]
    unique_filled = np.unique(filled_labels)
    print(f"Imputed label values: {unique_filled}")
    
    # Check if the number of samples per category is balanced
    if n_samples % n_classes == 0:
        for label in unique_labels:
            count = np.sum(imputed_labels == label)
            expected = samples_per_class
            if count != expected:
                print(f"Warning: Label {label} has {count} samples, expected {expected}.")
    
    return imputed_labels


def main():
    parser = argparse.ArgumentParser(description='Missing value imputation using subspace-aware Graph Neural Network.')
    parser.add_argument('--input', type=str, required=True, help='Input .mat file path.')
    parser.add_argument('--output_dir', type=str, default='data/datasets', help='Output directory, defaults to data/datasets.')
    parser.add_argument('--treat_zeros', action='store_true', help='Treat 0 values as missing values.')
    parser.add_argument('--k', type=int, default=10, help='Number of neighbors in KNN graph.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs.')
    parser.add_argument('--alpha', type=float, default=0.9, help='Weight coefficient for intra-subspace connections (0-1).')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for calculating similarity matrix in batches.')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA acceleration.')
    
    args = parser.parse_args()
    
    # Handle CUDA option
    if args.no_cuda:
        global device
        device = torch.device('cpu')
        print("CUDA disabled, using CPU.")
    
    # Print usage example
    if not os.path.exists(args.input):
        print(f"Error: File '{args.input}' does not exist.")
        print("\nUsage example:")
        print("  python subspace_aware_gnn.py --input data/datasets/COIL100_random_zero.mat --treat_zeros")
        print("  python subspace_aware_gnn.py --input data/datasets/ORL_32x32_random_zero.mat --k 15 --epochs 300 --alpha 0.9")
        print("\nAvailable dataset files:")
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
