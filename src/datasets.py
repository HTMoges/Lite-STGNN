#!/usr/bin/env python3
"""
Dataset Classes for Lite-STGNN
===============================

Dataset loaders for all models wich contains ElectricityDataset, TrafficDataset, ExchangeDataset and WeatherDataset classes.

Key Features:
- TimeLinear data loading pattern
- Standardized preprocessing across all models
- Consistent train/val/test splits
- Scalable to different sequence/prediction lengths
"""

import os
from typing import Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class ElectricityDataset(Dataset):
    """TimeLinear-style Electricity dataset loader for fair comparison"""
    
    def __init__(self, root_path, flag='train', seq_len=96, pred_len=720, features='M', scale=True):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = features
        self.scale = scale
        
        # TimeLinear split ratios
        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.root_path = root_path
        self.__read_data__()
    
    def __read_data__(self):
        """Load and preprocess electricity data"""
        # Load electricity.csv (same as TimeLinear)
        df_raw = pd.read_csv(os.path.join(self.root_path, 'electricity.csv'))
        
        # Remove date column if present, keep only numeric data
        if 'date' in df_raw.columns:
            df_data = df_raw.drop(['date'], axis=1)
        else:
            df_data = df_raw
        
        # TimeLinear split: 70% train, 20% test, 10% val
        num_train = int(len(df_data) * 0.7)
        num_test = int(len(df_data) * 0.2)
        num_val = len(df_data) - num_train - num_test
        
        # Define borders for train/val/test
        border1s = [0, num_train - self.seq_len, len(df_data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, len(df_data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # Scaling
        if self.scale:
            self.scaler = StandardScaler()
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            self.scaler = None
        
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        
        print(f"[Electricity {['train', 'val', 'test'][self.set_type]}] Data shape: {self.data_x.shape}")
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        
        return torch.FloatTensor(seq_x), torch.FloatTensor(seq_y)


class TrafficDataset(Dataset):
    """TimeLinear-style Traffic dataset loader for fair comparison"""
    
    def __init__(self, root_path, flag='train', seq_len=96, pred_len=720, features='M', scale=True):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = features
        self.scale = scale
        
        # TimeLinear split ratios
        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.root_path = root_path
        self.__read_data__()
    
    def __read_data__(self):
        """Load and preprocess traffic data"""
        # Load traffic.csv (same as TimeLinear)
        df_raw = pd.read_csv(os.path.join(self.root_path, 'traffic.csv'))
        
        # Remove date column if present, keep only numeric data
        if 'date' in df_raw.columns:
            df_data = df_raw.drop(['date'], axis=1)
        else:
            df_data = df_raw
        
        # TimeLinear split: 70% train, 20% test, 10% val
        num_train = int(len(df_data) * 0.7)
        num_test = int(len(df_data) * 0.2)
        num_val = len(df_data) - num_train - num_test
        
        # Define borders for train/val/test
        border1s = [0, num_train - self.seq_len, len(df_data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, len(df_data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # Scaling
        if self.scale:
            self.scaler = StandardScaler()
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            self.scaler = None
        
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        
        print(f"[Traffic {['train', 'val', 'test'][self.set_type]}] Data shape: {self.data_x.shape}")
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        
        return torch.FloatTensor(seq_x), torch.FloatTensor(seq_y)


class WeatherDataset(Dataset):
    """TimeLinear-style Weather dataset loader for fair comparison"""

    def __init__(self, root_path, flag='train', seq_len=96, pred_len=720, features='M', scale=True):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = features
        self.scale = scale

        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, 'weather.csv'))

        if 'date' in df_raw.columns:
            df_data = df_raw.drop(['date'], axis=1)
        else:
            df_data = df_raw

        num_train = int(len(df_data) * 0.7)
        num_test = int(len(df_data) * 0.2)
        num_val = len(df_data) - num_train - num_test

        border1s = [0, num_train - self.seq_len, len(df_data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, len(df_data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.scale:
            self.scaler = StandardScaler()
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            self.scaler = None

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        print(f"[Weather {['train', 'val', 'test'][self.set_type]}] Data shape: {self.data_x.shape}")

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return torch.FloatTensor(seq_x), torch.FloatTensor(seq_y)


class ExchangeDataset(Dataset):
    """TimeLinear-style Exchange Rate dataset loader for fair comparison"""

    def __init__(self, root_path, flag='train', seq_len=96, pred_len=720, features='M', scale=True):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = features
        self.scale = scale

        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.__read_data__()

    def __read_data__(self):
        # Some repos name it exchange_rate.csv; accept both
        csv_path = 'exchange_rate.csv'
        path_try = os.path.join(self.root_path, csv_path)
        if not os.path.exists(path_try):
            csv_path = 'exchange.csv'
        df_raw = pd.read_csv(os.path.join(self.root_path, csv_path))

        if 'date' in df_raw.columns:
            df_data = df_raw.drop(['date'], axis=1)
        else:
            df_data = df_raw

        num_train = int(len(df_data) * 0.7)
        num_test = int(len(df_data) * 0.2)
        num_val = len(df_data) - num_train - num_test

        border1s = [0, num_train - self.seq_len, len(df_data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, len(df_data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.scale:
            self.scaler = StandardScaler()
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            self.scaler = None

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        print(f"[Exchange {['train', 'val', 'test'][self.set_type]}] Data shape: {self.data_x.shape}")

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return torch.FloatTensor(seq_x), torch.FloatTensor(seq_y)


def get_dataset_info(dataset_name: str) -> dict:
    """Get dataset-specific configuration information"""
    
    if dataset_name.lower() == 'electricity':
        return {
            'name': 'electricity',
            'num_features': 321,
            'filename': 'electricity.csv',
            'description': 'Electricity Transformer Temperature (ETT) dataset with 321 features',
        }
    elif dataset_name.lower() == 'traffic':
        return {
            'name': 'traffic',
            'num_features': 862,
            'filename': 'traffic.csv', 
            'description': 'Traffic dataset with 862 sensor readings',
        }
    elif dataset_name.lower() == 'weather':
        return {
            'name': 'weather',
            'num_features': 21,  # common Weather multivariate feature count
            'filename': 'weather.csv',
            'description': 'Weather dataset with ~21 meteorological variables',
        }
    elif dataset_name.lower() in ['exchange', 'exchange_rate', 'exchange-rate']:
        return {
            'name': 'exchange_rate',
            'num_features': 8,  # common Exchange rate dataset variable count
            'filename': 'exchange_rate.csv',
            'description': 'Exchange Rate dataset with 8 currency series',
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def create_dataloaders(dataset_name: str, data_root: str, seq_len: int = 96, 
                      pred_len: int = 720, batch_size: Optional[int] = None, 
                      num_workers: int = 4, pin_memory: bool = True):
    """Create train/val/test dataloaders for specified dataset"""
    
    # Get dataset class
    name = dataset_name.lower()
    if name == 'electricity':
        dataset_class = ElectricityDataset
        default_batch_size = 16
    elif name == 'traffic':
        dataset_class = TrafficDataset
        default_batch_size = 8  # Smaller for memory efficiency
    elif name == 'weather':
        dataset_class = WeatherDataset
        default_batch_size = 16
    elif name in ['exchange', 'exchange_rate', 'exchange-rate']:
        dataset_class = ExchangeDataset
        default_batch_size = 16
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if batch_size is None:
        batch_size = default_batch_size
    
    # Create datasets
    train_dataset = dataset_class(data_root, flag='train', seq_len=seq_len, pred_len=pred_len)
    val_dataset = dataset_class(data_root, flag='val', seq_len=seq_len, pred_len=pred_len)
    test_dataset = dataset_class(data_root, flag='test', seq_len=seq_len, pred_len=pred_len)
    
    # Create dataloaders
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, 
        persistent_workers=(num_workers > 0)
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )
    
    return train_loader, val_loader, test_loader, (train_dataset, val_dataset, test_dataset)