"""Módulo para entrenamiento y búsqueda de hiperparámetros."""

import itertools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from models.network import ShallowCognitiveNet
from evaluation import apply_threshold, hamming_loss
from core.config import HYPERPARAMETER_SPACE
import random

def get_device() -> torch.device:
    """Retorna el dispositivo disponible (CUDA si hay GPU, sino CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    pos_weights: torch.Tensor,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device = None
) -> float:
    """
    Entrena el modelo y retorna la pérdida de validación (usaremos la loss
    para la selección de hiperparámetros por simplicidad y suavidad).
    """
    if device is None:
        device = get_device()
        
    model = model.to(device)
    pos_weights = pos_weights.to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validación
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
    return best_val_loss

def tune_hyperparameters(
    X_train_inner: np.ndarray,
    Y_train_inner: np.ndarray,
    inner_splits: list,
    pos_weights_np: np.ndarray,
    input_dim: int,
    num_classes: int,
    epochs: int = 20,
    activation_name: str = 'Mish',
    n_iter: int = 10
) -> dict:
    """
    Realiza Random Search sobre los Folds Internos para encontrar los mejores
    hiperparámetros explorando subconjuntos aleatorios del espacio.
    """
    device = get_device()
    pos_weights = torch.tensor(pos_weights_np, dtype=torch.float32)
    
    space = HYPERPARAMETER_SPACE
    keys = list(space.keys())
    
    # Generar iteraciones aleatorias de hiperparámetros
    hyperparameter_combinations = []
    for _ in range(n_iter):
        params = {k: random.choice(space[k]) for k in keys}
        if params not in hyperparameter_combinations: # Evitar duplicados si es posible
            hyperparameter_combinations.append(params)
    
    best_params = None
    best_avg_loss = float('inf')
    
    from data_loader import CognitiveMultiLabelDataset
    
    for params in hyperparameter_combinations:
        fold_losses = []
        
        for train_idx, val_idx in inner_splits:
            X_t, Y_t = X_train_inner[train_idx], Y_train_inner[train_idx]
            X_v, Y_v = X_train_inner[val_idx], Y_train_inner[val_idx]
            
            # Imputar NaNs y escalar internamente para evitar leakage en inner folds
            from preprocessing import impute_and_scale_features
            X_t_scaled, X_v_scaled = impute_and_scale_features(X_t, X_v)
            
            train_dataset = CognitiveMultiLabelDataset(X_t_scaled, Y_t)
            val_dataset = CognitiveMultiLabelDataset(X_v_scaled, Y_v)
            
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
            
            model = ShallowCognitiveNet(
                input_dim=input_dim, 
                num_classes=num_classes, 
                hidden_dim=params['hidden_dim'],
                dropout_prob=params['dropout_prob'],
                activation_name=activation_name
            )
            
            val_loss = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                pos_weights=pos_weights,
                epochs=epochs,
                lr=params['lr'],
                weight_decay=params['weight_decay'],
                device=device
            )
            fold_losses.append(val_loss)
            
        avg_loss = np.mean(fold_losses)
        if avg_loss < best_avg_loss:
            best_avg_loss = avg_loss
            best_params = params
            
    return best_params
