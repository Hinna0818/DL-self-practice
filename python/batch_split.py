## 多种方法实现数据集的批次分割(batch split)
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

## 方法1：使用PyTorch DataLoader进行分割
def batch_split_torch(X, y, batch_size = 16, shuffle = True):
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)
    for X_batch, y_batch in dataloader:
        print(X_batch, y_batch)


## 方法2：使用numpy切片进行分割
def batch_split_np(X, y, batch_size = 16, shuffle = True):
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
    
    X = X[indices]
    y = y[indices]
    num_batches = n_samples // batch_size
    for i in range(num_batches):
        X_batch = X[i * batch_size : (i + 1) * batch_size]
        y_batch = y[i * batch_size : (i + 1) * batch_size]
        print(X_batch, y_batch)

## 方法3：使用生成器进行迭代式批处理
def batch_generator(X, y, batch_size = 16, shuffle = True):
    num_samples = len(X)
    if shuffle:
        indices = np.arange(num_samples)
        np.random.shuffle(indices)  
    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i : i + batch_size]
        yield X[batch_indices], y[batch_indices]
