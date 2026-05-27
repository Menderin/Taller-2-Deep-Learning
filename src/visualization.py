"""Módulo para visualizaciones de datos y resultados (Análisis Exploratorio y Preprocesamiento)."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import os

def plot_class_distribution(Y_encoded: np.ndarray, target_name: str, classes: list, output_dir: str = "results/plots"):
    """
    Genera un gráfico de barras mostrando el desbalanceo de clases del target.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Contar positivos por cada clase
    counts = Y_encoded.sum(axis=0)
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=[str(c) for c in classes], y=counts, palette="viridis")
    plt.title(f"Distribución de Clases para {target_name}")
    plt.xlabel("Clases")
    plt.ylabel("Cantidad de Muestras")
    
    # Añadir el número exacto encima de cada barra
    for i, count in enumerate(counts):
        plt.text(i, count + (counts.max() * 0.01), str(int(count)), ha='center')
        
    plt.tight_layout()
    plt.savefig(f"{output_dir}/distribucion_{target_name}.png", dpi=300)
    plt.close()

def plot_scaling_comparison(X_raw: np.ndarray, X_scaled: np.ndarray, feature_idx: int = 0, feature_name: str = "Feature", output_dir: str = "results/plots"):
    """
    Compara la distribución de una variable antes y después de aplicar StandardScaler.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Antes del escalamiento
    sns.histplot(X_raw[:, feature_idx], kde=True, ax=axes[0], color='coral')
    axes[0].set_title(f"Distribución Original: {feature_name}")
    axes[0].set_xlabel("Valor")
    axes[0].set_ylabel("Frecuencia")
    
    # Después del escalamiento
    sns.histplot(X_scaled[:, feature_idx], kde=True, ax=axes[1], color='teal')
    axes[1].set_title(f"Distribución Escalada (StandardScaler)")
    axes[1].set_xlabel("Z-Score")
    axes[1].set_ylabel("Frecuencia")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/escalamiento_comparacion.png", dpi=300)
    plt.close()

def plot_uncertainty_distribution(uncertainty_vars: list, target_name: str, activation_name: str, output_dir: str = "results/plots"):
    """
    Visualiza la distribución de la varianza epistémica (incertidumbre) capturada por MC Dropout.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(8, 5))
    sns.histplot(uncertainty_vars, kde=True, color='purple', bins=30)
    plt.title(f"Incertidumbre (MC Dropout Var) | {target_name} | {activation_name}")
    plt.xlabel("Varianza Promedio (Incertidumbre Epistémica)")
    plt.ylabel("Frecuencia (En lotes/batches)")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mc_dropout_uncertainty_{target_name}.png", dpi=300)
    plt.close()

def plot_uncertainty_vs_probability(mean_probs: np.ndarray, std_probs: np.ndarray, target_name: str, activation_name: str, output_dir: str = "results/plots"):
    """
    Scatter plot que relaciona la Probabilidad Media frente a su Desviación Estándar (Incertidumbre).
    Permite validar visualmente si las probabilidades cerca del umbral de decisión (0.5) 
    tienen mayor incertidumbre (comportamiento esperado).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Aplanamos los arreglos para poder graficar punto a punto sin importar la dimensionalidad de salida (clases)
    plt.figure(figsize=(8, 6))
    
    # Podría haber muchas muestras, bajamos el alpha para no saturar 
    plt.scatter(mean_probs.flatten(), std_probs.flatten(), alpha=0.3, color='crimson')
    
    # Dibujar la línea de umbral
    plt.axvline(x=0.5, color='black', linestyle='--', label='Umbral de decisión (0.5)')
    
    plt.title(f"Probabilidad vs Incertidumbre (MC Dropout) | {target_name} | {activation_name}")
    plt.xlabel("Probabilidad Media Observada")
    plt.ylabel("Desviación Estándar (Incertidumbre)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mc_dropout_prob_vs_std_{target_name}.png", dpi=300)
    plt.close()

def plot_learning_curves(history: dict, target_name: str, activation_name: str, output_dir: str = "results/plots"):
    """
    Genera el gráfico de Curvas de Aprendizaje (Loss) por épocas.
    Visualiza si el modelo está convergiendo bien o si está sufriendo de overfitting.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    plt.plot(epochs, history['val_loss'], label='Validation/Test Loss', color='orange', linewidth=2, linestyle='--')
    
    plt.title(f"Curva de Aprendizaje | {target_name} | {activation_name}")
    plt.xlabel("Épocas (Epochs)")
    plt.ylabel("BCE Loss (Log-Loss)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/learning_curve_{target_name}.png", dpi=300)
    plt.close()
