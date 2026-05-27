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
from tqdm import tqdm

def get_device() -> torch.device | list[int]:
    """Retorna el dispositivo disponible. Si hay múltiples GPUs, se preparará para DataParallel."""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"Detectadas {num_gpus} GPUs. Usaremos DataParallel/Multi-GPU.")
            return list(range(num_gpus))
        return torch.device("cuda:0")
    return torch.device("cpu")

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    pos_weights: torch.Tensor,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | list[int] = None,
    silent: bool = False
) -> tuple[float, nn.Module]:
    """
    Entrena el modelo y retorna la pérdida de validación y el modelo.
    """
    if device is None:
        device = get_device()
        
    multi_gpu = isinstance(device, list)
    
    if multi_gpu:
        active_device = torch.device(f"cuda:{device[0]}")
        model = model.to(active_device)
        model = nn.DataParallel(model, device_ids=device)
        pos_weights = pos_weights.to(active_device)
    else:
        active_device = device
        model = model.to(active_device)
        pos_weights = pos_weights.to(active_device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    iterator = range(epochs) if silent else tqdm(range(epochs), desc=f"Entrenando", leave=False)
    for epoch in iterator:
        model.train()
        train_loss = 0.0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(active_device), Y_batch.to(active_device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
            
        # Validación
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(active_device), Y_batch.to(active_device)
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
    # Extraer el módulo interno si usamos DataParallel
    if multi_gpu:
        model = model.module
            
    return best_val_loss, model, history

def _evaluate_hyperparameter(params, X_train_inner, Y_train_inner, inner_splits, pos_weights_np, input_dim, num_classes, epochs, activation_name):
    """Función auxiliar para evaluar una combinación de hiperparámetros de manera paralela."""
    import torch
    
    # CRÍTICO: Prevenir la explosión de hilos cuando usamos joblib.Parallel
    torch.set_num_threads(1)
    
    import numpy as np
    from torch.utils.data import DataLoader
    from data_loader import CognitiveMultiLabelDataset
    from models.network import ShallowCognitiveNet
    from preprocessing import impute_and_scale_features
    # Forzamos CPU para las evaluaciones paralelas internas
    device = torch.device('cpu') 
    pos_weights = torch.tensor(pos_weights_np, dtype=torch.float32)
    
    fold_losses = []
    
    for train_idx, val_idx in inner_splits:
        X_t, Y_t = X_train_inner[train_idx], Y_train_inner[train_idx]
        X_v, Y_v = X_train_inner[val_idx], Y_train_inner[val_idx]
        
        # Imputar NaNs y escalar internamente para evitar leakage en inner folds
        X_t_scaled, X_v_scaled = impute_and_scale_features(X_t, X_v)
        
        train_dataset = CognitiveMultiLabelDataset(X_t_scaled, Y_t)
        val_dataset = CognitiveMultiLabelDataset(X_v_scaled, Y_v)
        
        # Uso de workers no es necesario acá ya que la RAM ya tiene los tensores en CPU
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
        
        model = ShallowCognitiveNet(
            input_dim=input_dim, 
            num_classes=num_classes, 
            hidden_dim=params['hidden_dim'],
            dropout_prob=params['dropout_prob'],
            activation_name=activation_name
        )
        
        val_loss, _, _ = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            pos_weights=pos_weights,
            epochs=epochs,
            lr=params['lr'],
            weight_decay=params['weight_decay'],
            device=device,
            silent=True
        )
        fold_losses.append(val_loss)
        
    return np.mean(fold_losses)

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
    
    print(f"Probando {len(hyperparameter_combinations)} combinaciones en paralelo (Random Search)...")
    
    from joblib import Parallel, delayed
    import os
    
    # Paralelizamos limitando la cantidad de procesos para no acaparar en exceso el servidor
    n_jobs = min(16, os.cpu_count() or 1)
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_hyperparameter)(
            params, X_train_inner, Y_train_inner, inner_splits, 
            pos_weights_np, input_dim, num_classes, epochs, activation_name
        ) for params in tqdm(hyperparameter_combinations, desc="Lanzando Workers")
    )
    
    # Encontrar la mejor combinación
    best_idx = np.argmin(results)
    best_avg_loss = results[best_idx]
    best_params = hyperparameter_combinations[best_idx]
    
    print(f"Mejor Avg Loss Interna: {best_avg_loss:.4f} con {best_params}")
    return best_params
