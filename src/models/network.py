"""Arquitectura de la Red Neuronal PyTorch."""

import torch
import torch.nn as nn

class ShallowCognitiveNet(nn.Module):
    """
    Red neuronal poco profunda para la predicción de deterioro cognitivo.
    Diseñada para ser simple y didáctica para el Laboratorio 02.
    """
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 32, dropout_prob: float = 0.3, activation_name: str = 'Mish'):
        super(ShallowCognitiveNet, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        if activation_name.lower() == 'relu':
            activation_layer = nn.ReLU()
        elif activation_name.lower() == 'mish':
            activation_layer = nn.Mish()
        else:
            raise ValueError(f"Activación no soportada: {activation_name}")
            
        # Arquitectura simple: 1 capa oculta
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation_layer,
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Paso forward de la red.
        No incluye Sigmoid ya que usaremos BCEWithLogitsLoss que lo combina
        por razones de estabilidad numérica.
        """
        return self.network(x)
