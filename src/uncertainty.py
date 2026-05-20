"""Interfaces para la parte de incertidumbre del laboratorio."""

import torch
import torch.nn as nn


def enable_dropout_during_inference(model: nn.Module) -> None:
    """
    Activa las capas Dropout durante inferencia para poder usar Monte Carlo Dropout.
    Itera sobre todos los módulos de la red y si es una capa Dropout, la pone en modo train.
    """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def mc_dropout_predict(
    model: nn.Module, inputs: torch.Tensor, n_samples: int = 30
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Repetir varias predicciones con dropout activo y resumir:
    - media de prediccion (probabilidades),
    - medida de incertidumbre (varianza).
    """
    model.eval()  # Asegurar que BatchNorm y otras capas estén en eval
    enable_dropout_during_inference(model)  # Solo activar Dropout
    
    predictions = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            # Obtener logits
            logits = model(inputs)
            # Convertir a probabilidades usando sigmoide (ya que usamos BCEWithLogitsLoss)
            probs = torch.sigmoid(logits)
            predictions.append(probs)
            
    # Convertir a tensor 3D: (n_samples, batch_size, num_classes)
    predictions = torch.stack(predictions)
    
    # Calcular media y varianza a través de la dimensión de las muestras
    mean_probs = predictions.mean(dim=0)
    variance_probs = predictions.var(dim=0)
    
    return mean_probs, variance_probs