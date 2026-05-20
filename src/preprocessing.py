"""Funciones para preparar el target multilabel del experimento activo."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold

from core.config import TARGET_COLUMNS
from data_loader import build_input_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def encode_target_as_one_hot(
    dataframe: pd.DataFrame, target_name: str
) -> tuple[np.ndarray, list[int], dict[int, int]]:
    """
    Toma una sola columna objetivo y la convierte a formato one-hot.

    Ejemplo:
    Si la clase original es 2 y el experimento tiene tres clases,
    entonces el vector queda como [0, 1, 0].
    """

    if target_name not in TARGET_COLUMNS:
        raise ValueError(
            f"Target invalido: {target_name}. Debe ser uno de {TARGET_COLUMNS}."
        )

    if target_name not in dataframe.columns:
        raise ValueError(f"La columna objetivo {target_name} no existe en el dataset.")

    y_raw = dataframe[target_name].astype(int)
    classes = sorted(y_raw.unique().tolist())
    class_to_idx = {class_value: idx for idx, class_value in enumerate(classes)}

    y_idx = y_raw.map(class_to_idx).to_numpy()
    y_encoded = np.zeros((len(y_idx), len(classes)), dtype=np.float32)
    y_encoded[np.arange(len(y_idx)), y_idx] = 1.0

    return y_encoded, classes, class_to_idx


def prepare_experiment_data(
    dataframe: pd.DataFrame, target_name: str
) -> tuple[np.ndarray, np.ndarray, list[int], dict[int, int]]:
    """Prepara X e Y para un experimento puntual."""

    X = build_input_matrix(dataframe)
    Y, classes, class_to_idx = encode_target_as_one_hot(dataframe, target_name)
    return X, Y, classes, class_to_idx


def compute_class_weights(Y_train: np.ndarray) -> np.ndarray:
    """
    Calcula los pesos positivos (pos_weight) para BCEWithLogitsLoss.
    Fórmula: pos_weight = num_negativos / num_positivos para cada clase.
    """
    Y_train_tensor = np.asarray(Y_train, dtype=np.float32)
    positives = Y_train_tensor.sum(axis=0)
    negatives = len(Y_train_tensor) - positives
    
    # Evitar division por cero añadiendo un pequeño epsilon
    pos_weights = negatives / (positives + 1e-7)
    return pos_weights.astype(np.float32)


def impute_and_scale_features(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray = None) -> tuple:
    """
    Maneja valores NaN imputando con la moda (most_frequent) dado que los atributos son binarios.
    Luego ajusta un StandardScaler solo en entrenamiento y transforma los demás para evitar data leakage.
    """
    imputer = SimpleImputer(strategy='most_frequent')
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_val_scaled = scaler.transform(X_val_imputed)
    
    if X_test is not None:
        X_test_imputed = imputer.transform(X_test)
        X_test_scaled = scaler.transform(X_test_imputed)
        return X_train_scaled, X_val_scaled, X_test_scaled
        
    return X_train_scaled, X_val_scaled


def split_for_validation(
    Y: np.ndarray, n_splits: int, random_state: int = 42, target_name: str = "Target"
) -> list[tuple[np.ndarray, np.ndarray]]:
    """API publica para crear folds estratificados del laboratorio."""

    return build_nested_splits(Y, n_splits=n_splits, random_state=random_state, target_name=target_name)


def build_stratification_labels(Y: np.ndarray) -> np.ndarray:
    """
    Construye etiquetas auxiliares para estratificar los folds.

    En esta version minima del laboratorio, cada experimento toma una sola
    columna objetivo y la codifica como one-hot. Eso permite estratificar
    usando la clase activa de cada fila.
    """

    Y = np.asarray(Y, dtype=np.int64)

    if Y.ndim != 2:
        raise ValueError("Y debe ser una matriz 2D para construir folds estratificados.")

    label_counts = Y.sum(axis=1)
    if np.any(label_counts <= 0):
        raise ValueError("Cada muestra debe activar al menos una etiqueta en Y.")

    if np.all(label_counts == 1):
        return np.argmax(Y, axis=1)

    return np.asarray(["|".join(map(str, row.tolist())) for row in Y], dtype=object)


def build_nested_splits(
    Y: np.ndarray, n_splits: int, random_state: int = 42, target_name: str = "Target"
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Genera folds estratificados para la validacion del laboratorio.

    Si luego el curso adopta una estrategia mas avanzada de estratificacion
    multilabel, este es el lugar natural para reemplazar la implementacion.
    """

    labels = build_stratification_labels(Y)
    unique_labels, counts = np.unique(labels, return_counts=True)

    if len(unique_labels) < 2:
        raise ValueError(
            "Se requieren al menos dos clases distintas para realizar validacion."
        )

    if counts.min() < n_splits:
        print(f"[{target_name}] ADVERTENCIA: La clase menos frecuente solo tiene {counts.min()} muestras. "
              f"Se usará KFold estándar (sin estratificar) para {n_splits} splits para evitar fallos.")
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        dummy_inputs = np.zeros(len(labels), dtype=np.float32)
        return list(splitter.split(dummy_inputs))

    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    dummy_inputs = np.zeros(len(labels), dtype=np.float32)
    return list(splitter.split(dummy_inputs, labels))