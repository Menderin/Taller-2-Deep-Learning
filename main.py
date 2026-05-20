"""Orquestador Principal del Laboratorio 02."""

import sys
from pathlib import Path
import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Añadir src al path para importar módulos locales fácilmente
sys.path.append(str(Path(__file__).parent / "src"))

from core.config import (
    TARGET_COLUMNS, 
    DEFAULT_OUTER_FOLDS, 
    DEFAULT_INNER_FOLDS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    ACTIVATION_FUNCTIONS
)
from data_loader import load_dataframe, CognitiveMultiLabelDataset
from preprocessing import (
    prepare_experiment_data,
    build_nested_splits,
    compute_class_weights,
    impute_and_scale_features
)
from models.network import ShallowCognitiveNet
from training import tune_hyperparameters, train_model, get_device
from evaluation import (
    apply_threshold, 
    hamming_loss, 
    exact_match_accuracy, 
    f1_multilabel,
    precision_multilabel,
    recall_multilabel
)
from uncertainty import mc_dropout_predict
from visualization import plot_class_distribution, plot_scaling_comparison, plot_uncertainty_distribution, plot_uncertainty_vs_probability

def run_experiment(dataframe: pd.DataFrame, target_name: str, activation_name: str = 'Mish') -> dict:
    """Ejecuta el experimento completo (Nested CV) para un target específico."""
    print(f"\n{'='*50}\nIniciando Experimento para Target: {target_name} | Activación: {activation_name}\n{'='*50}")
    
    # 1. Preparación de datos (Preprocesamiento)
    X, Y, classes, class_to_idx = prepare_experiment_data(dataframe, target_name)
    input_dim = X.shape[1]
    num_classes = Y.shape[1]
    
    # Crear directorio para este target y activación
    target_dir = Path(f"results/{activation_name}/{target_name}")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualización Pedagógica: Desbalanceo de Clases
    plot_class_distribution(Y, target_name, classes, output_dir=str(target_dir))
    
    # 2. Partición Externa (Outer Folds para Evaluación)
    outer_splits = build_nested_splits(Y, n_splits=DEFAULT_OUTER_FOLDS)
    
    experiment_results = {
        'hamming_loss': [],
        'exact_match': [],
        'precision_micro': [],
        'recall_micro': [],
        'f1_micro': [],
        'f1_macro': [],
        'mc_uncertainty_mean_var': []
    }
    
    for outer_idx, (train_outer_idx, test_idx) in enumerate(outer_splits):
        print(f"\n--- Outer Fold {outer_idx + 1}/{DEFAULT_OUTER_FOLDS} ---")
        
        X_train_outer, Y_train_outer = X[train_outer_idx], Y[train_outer_idx]
        X_test, Y_test = X[test_idx], Y[test_idx]
        
        # 3. Calcular class weights en train_outer
        pos_weights = compute_class_weights(Y_train_outer)
        
        # 4. Partición Interna (Inner Folds para Búsqueda de Hiperparámetros)
        inner_splits = build_nested_splits(Y_train_outer, n_splits=DEFAULT_INNER_FOLDS)
        
        print("Realizando Búsqueda de Hiperparámetros (Random Search)...")
        best_params = tune_hyperparameters(
            X_train_inner=X_train_outer,
            Y_train_inner=Y_train_outer,
            inner_splits=inner_splits,
            pos_weights_np=pos_weights,
            input_dim=input_dim,
            num_classes=num_classes,
            epochs=DEFAULT_EPOCHS,
            activation_name=activation_name,
            n_iter=10
        )
        print(f"Mejores parámetros encontrados: {best_params}")
        
        # 5. Imputar NaNs y Escalar datos (Fit en train_outer, Transform en test)
        X_train_scaled, _, X_test_scaled = impute_and_scale_features(X_train_outer, X_train_outer, X_test)
        
        # Visualización Pedagógica: Comparación Escalamiento (Solo en el primer outer fold del primer target)
        if outer_idx == 0:
            from core.config import FEATURE_COLUMNS
            plot_scaling_comparison(X_train_outer, X_train_scaled, feature_idx=0, feature_name=FEATURE_COLUMNS[0], output_dir=str(target_dir))
        
        # Datasets y Loaders
        train_dataset = CognitiveMultiLabelDataset(X_train_scaled, Y_train_outer)
        test_dataset = CognitiveMultiLabelDataset(X_test_scaled, Y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
        # Para inferencia no mezclamos
        test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)
        
        # 6. Entrenar el Mejor Modelo Final sobre todo el Outer Fold
        model = ShallowCognitiveNet(
            input_dim=input_dim, 
            num_classes=num_classes, 
            hidden_dim=best_params['hidden_dim'],
            dropout_prob=best_params['dropout_prob'],
            activation_name=activation_name
        )
        
        device = get_device()
        
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,  # Usamos test como validación para que el loop funcione, pero evaluamos después
            pos_weights=torch.tensor(pos_weights, dtype=torch.float32),
            epochs=DEFAULT_EPOCHS,
            lr=best_params['lr'],
            weight_decay=best_params['weight_decay'],
            device=device
        )
        
        # 7. Evaluación con Métricas y Monte Carlo Dropout
        all_y_true = []
        all_y_pred = []
        all_mean_probs = []
        all_std_probs = []
        uncertainty_variances = []
        
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.cpu().numpy()
            
            # Estimación de Incertidumbre usando MC Dropout (N=30)
            mean_probs, variance_probs = mc_dropout_predict(model, X_batch, n_samples=30)
            
            mean_probs_np = mean_probs.cpu().numpy()
            y_pred_batch = apply_threshold(mean_probs_np, threshold=0.5)
            std_probs_np = np.sqrt(variance_probs.cpu().numpy()) # Desviación Estándar para los plots
            
            all_y_true.append(Y_batch)
            all_y_pred.append(y_pred_batch)
            
            all_mean_probs.append(mean_probs_np)
            all_std_probs.append(std_probs_np)
            uncertainty_variances.append(variance_probs.mean().item()) # Varianza promedio del batch
            
        y_true_full = np.vstack(all_y_true)
        y_pred_full = np.vstack(all_y_pred)
        
        # Calcular métricas
        hl = hamming_loss(y_true_full, y_pred_full)
        em = exact_match_accuracy(y_true_full, y_pred_full)
        prec_mic = precision_multilabel(y_true_full, y_pred_full, average='micro')
        rec_mic = recall_multilabel(y_true_full, y_pred_full, average='micro')
        f1_mic = f1_multilabel(y_true_full, y_pred_full, average='micro')
        f1_mac = f1_multilabel(y_true_full, y_pred_full, average='macro')
        avg_uncertainty = np.mean(uncertainty_variances)
        
        # Visualización Pedagógica de Incertidumbre
        if outer_idx == 0:
            full_mean_probs = np.vstack(all_mean_probs)
            full_std_probs = np.vstack(all_std_probs)
            
            plot_uncertainty_distribution(uncertainty_variances, target_name, activation_name, output_dir=str(target_dir))
            plot_uncertainty_vs_probability(full_mean_probs, full_std_probs, target_name, activation_name, output_dir=str(target_dir))
            
        print(f"Resultados Outer Fold {outer_idx + 1}:")
        print(f"Hamming Loss: {hl:.4f} | Exact Match: {em:.4f} | F1 (Micro): {f1_mic:.4f} | Precision (Micro): {prec_mic:.4f} | Recall (Micro): {rec_mic:.4f}")
        
        experiment_results['hamming_loss'].append(hl)
        experiment_results['exact_match'].append(em)
        experiment_results['precision_micro'].append(prec_mic)
        experiment_results['recall_micro'].append(rec_mic)
        experiment_results['f1_micro'].append(f1_mic)
        experiment_results['f1_macro'].append(f1_mac)
        experiment_results['mc_uncertainty_mean_var'].append(avg_uncertainty)
        
        # Guardar los pesos del modelo para este fold
        torch.save(model.state_dict(), target_dir / f"best_model_fold_{outer_idx + 1}.pth")
        
    # Promedios del experimento
    summary = {k: float(np.mean(v)) for k, v in experiment_results.items()}
    summary['target'] = target_name
    summary['activation'] = activation_name
    # Añadimos la mejor configuración encontrada en el último fold para el análisis del estudiante
    summary['best_hidden_dim_fold5'] = best_params['hidden_dim']
    summary['best_lr_fold5'] = best_params['lr']
    summary['best_dropout_fold5'] = best_params['dropout_prob']
    
    print(f"\n[RESUMEN {target_name} | {activation_name}] -> {summary}")
    return summary


def main():
    print("Iniciando Pipeline de Laboratorio 02 - Deep Learning")
    device = get_device()
    print(f"Dispositivo activo: {device}")
    
    # Asegurar que el directorio de resultados exista
    Path("results").mkdir(exist_ok=True)
    
    from data_loader import load_dataframe, CognitiveMultiLabelDataset, prepare_sav_to_csv
    
    # 1. Rutas de los datos
    raw_data_path = Path("data/raw/15 atributos R0-R5.sav")
    proc_data_path = Path("data/processed/dataset.csv")
    
    # 2. Pipeline de Preprocesamiento automágico
    if proc_data_path.exists():
        print(f"Detectado dataset procesado en {proc_data_path}. Cargándolo directamente...")
        df = load_dataframe(proc_data_path)
    elif raw_data_path.exists():
        print(f"Dataset procesado no encontrado. Extrayendo datos desde el original: {raw_data_path}...")
        df = prepare_sav_to_csv(raw_data_path, proc_data_path)
    else:
        print(f"¡ADVERTENCIA CRÍTICA! No se encontró ni {proc_data_path} ni {raw_data_path}.")
        print("Generando Dummy Dataset binario para demostración (Solo para propósitos de Testing).")
        proc_data_path.parent.mkdir(parents=True, exist_ok=True)
        from core.config import FEATURE_COLUMNS, TARGET_COLUMNS
        dummy_df = pd.DataFrame(np.random.choice([0, 1], size=(200, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
        for t in TARGET_COLUMNS:
            dummy_df[t] = np.random.randint(0, 3, size=200) # Ej: Clases 0, 1, 2
        dummy_df.to_csv(proc_data_path, index=False)
        df = load_dataframe(proc_data_path)
        
    print(f"Dataset cargado con éxito. Shape: {df.shape}")
    print("\nVisualizando las primeras 10 filas (Muestra de Datos):")
    print(df.head(10).to_string())
    print("-" * 60)
    
    all_summaries = []
    
    # 2. Ejecutar experimentos independientemente por target y por activación
    for activation in ACTIVATION_FUNCTIONS:
        print(f"\n{'*'*60}\nINICIANDO BATERÍA DE EXPERIMENTOS CON ACTIVACIÓN: {activation}\n{'*'*60}")
        for target in TARGET_COLUMNS:
            summary = run_experiment(df, target, activation_name=activation)
            all_summaries.append(summary)
        
    # 3. Guardar resultados
    results_df = pd.DataFrame(all_summaries)
    
    # Exportar como CSV para análisis de datos
    results_df.to_csv("results/comparison_summary.csv", index=False)
    
    # Exportar como JSON plano
    results_df.to_json("results/comparison_summary.json", orient="records", indent=4)
    
    # Generar tabla Markdown (Formato plano alineado a la rúbrica del laboratorio)
    md_columns = ['target', 'activation', 'f1_macro', 'f1_micro', 'hamming_loss']
    if all(c in results_df.columns for c in md_columns):
        md_df = results_df[md_columns].copy()
        md_df.rename(columns={
            'target': 'Etiqueta', 
            'activation': 'Activación', 
            'f1_macro': 'F1 Macro', 
            'f1_micro': 'F1 Micro', 
            'hamming_loss': 'Hamming Loss'
        }, inplace=True)
        # Añadir columna vacía de observación para que el alumno la llene
        md_df['Observación'] = "..." 
        with open("results/tabla_comparacion.md", "w", encoding="utf-8") as f:
            f.write("## Tabla Comparativa de Experimentos\n\n")
            f.write(md_df.to_markdown(index=False))
            
    print("\n¡Ejecución completada! Resultados guardados en 'results/' (CSV, JSON plano y Markdown).")

if __name__ == "__main__":
    main()