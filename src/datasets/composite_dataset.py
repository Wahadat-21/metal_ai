# src/datasets/composite_dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import json
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, LabelEncoder


@dataclass
class CompositeDataConfig:
    """Configuration for composite material data"""
    composition_features: list
    process_features: list
    property_targets: list
    categorical_features: list
    numerical_features: list
    normalization: bool = True


class CompositeDataset(Dataset):
    """Dataset for composite material properties"""
    
    def __init__(
        self,
        data_path: Path,
        config: CompositeDataConfig,
        split: str = "train",
        transform: Optional[callable] = None,
    ):
        self.data_path = Path(data_path)
        self.config = config
        self.split = split
        self.transform = transform
        
        # Load data
        self.data = self._load_data()
        self.features = self._prepare_features()
        self.targets = self._prepare_targets()
        
        # Initialize scalers
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
    def _load_data(self) -> pd.DataFrame:
        """Load composite material data"""
        file_path = self.data_path / f"{self.split}.csv"
        if not file_path.exists():
            # Generate synthetic data if real data not available
            return self._generate_synthetic_data()
        return pd.read_csv(file_path)
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic composite material data for demonstration"""
        np.random.seed(42)
        n_samples = 1000
        
        # Composition features (volume fractions of constituents)
        n_constituents = 5
        compositions = np.random.dirichlet(np.ones(n_constituents), size=n_samples)
        
        # Process parameters
        temperatures = np.random.uniform(100, 300, n_samples)
        pressures = np.random.uniform(1, 100, n_samples)
        curing_times = np.random.uniform(1, 10, n_samples)
        
        # Calculate properties (synthetic)
        # Young's modulus depends on composition and processing
        modulus = (
            50 * compositions[:, 0] + 
            200 * compositions[:, 1] + 
            10 * compositions[:, 2] + 
            80 * compositions[:, 3] + 
            30 * compositions[:, 4] +
            0.5 * temperatures / 100 +
            0.1 * pressures +
            0.5 * curing_times +
            np.random.normal(0, 5, n_samples)
        )
        
        # Tensile strength
        strength = (
            20 * compositions[:, 0] +
            150 * compositions[:, 1] +
            5 * compositions[:, 2] +
            60 * compositions[:, 3] +
            15 * compositions[:, 4] +
            0.3 * temperatures / 100 +
            0.05 * pressures +
            0.3 * curing_times +
            np.random.normal(0, 3, n_samples)
        )
        
        # Create DataFrame
        columns = (
            [f"constituent_{i}" for i in range(n_constituents)] +
            ["temperature", "pressure", "curing_time"] +
            ["modulus", "strength"]
        )
        
        data = np.hstack([compositions, 
                         temperatures.reshape(-1, 1),
                         pressures.reshape(-1, 1),
                         curing_times.reshape(-1, 1),
                         modulus.reshape(-1, 1),
                         strength.reshape(-1, 1)])
        
        df = pd.DataFrame(data, columns=columns)
        return df
    
    def _prepare_features(self) -> np.ndarray:
        """Prepare feature matrix"""
        feature_cols = (self.config.composition_features + 
                       self.config.process_features)
        
        # Handle categorical features
        for col in self.config.categorical_features:
            if col in self.data.columns:
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col].astype(str))
        
        features = self.data[feature_cols].values.astype(np.float32)
        
        if self.config.normalization and self.split == "train":
            features = self.feature_scaler.fit_transform(features)
        elif self.config.normalization:
            features = self.feature_scaler.transform(features)
            
        return features
    
    def _prepare_targets(self) -> np.ndarray:
        """Prepare target matrix"""
        targets = self.data[self.config.property_targets].values.astype(np.float32)
        
        if self.config.normalization and self.split == "train":
            targets = self.target_scaler.fit_transform(targets)
        elif self.config.normalization:
            targets = self.target_scaler.transform(targets)
            
        return targets
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        targets = torch.tensor(self.targets[idx], dtype=torch.float32)
        
        if self.transform:
            features, targets = self.transform(features, targets)
            
        return features, targets


def create_dataloaders(
    data_path: Path,
    config: CompositeDataConfig,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Dict[str, DataLoader]:
    """Create train, validation, and test dataloaders"""
    
    # Load and split data
    full_data = pd.read_csv(data_path / "raw" / "composite_data.csv")
    n = len(full_data)
    
    indices = np.random.permutation(n)
    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)
    
    splits = {
        "train": full_data.iloc[indices[:train_end]],
        "val": full_data.iloc[indices[train_end:val_end]],
        "test": full_data.iloc[indices[val_end:]],
    }
    
    dataloaders = {}
    
    for split_name, split_data in splits.items():
        # Save split data
        split_path = data_path / "processed" / f"{split_name}.csv"
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_data.to_csv(split_path, index=False)
        
        # Create dataset
        dataset = CompositeDataset(
            data_path=data_path / "processed",
            config=config,
            split=split_name,
        )
        
        # Create dataloader
        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split_name == "train"),
        )
    
    return dataloaders
