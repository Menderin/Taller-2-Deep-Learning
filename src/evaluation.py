"""Metricas para evaluar salidas multilabel."""

import numpy as np


def apply_threshold(probabilities: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convierte probabilidades a etiquetas binarias usando un umbral fijo."""

    return (probabilities >= threshold).astype(np.float32)


def hamming_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula Hamming loss.

    La idea es simple:
    - comparar cada componente del vector real con la predicha,
    - contar cuantas componentes quedan distintas,
    - y promediar ese error.
    """

    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true e y_pred deben tener la misma forma para calcular Hamming loss."
        )

    mistakes = np.not_equal(y_true, y_pred)
    return float(mistakes.mean())


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def exact_match_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula Exact Match Ratio (Subset Accuracy).
    Requiere que todas las etiquetas de una instancia sean predichas correctamente.
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(accuracy_score(y_true, y_pred))


def f1_multilabel(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> float:
    """
    Calcula el F1-Score para problemas multilabel.
    'macro' calcula el F1 por etiqueta y promedia (sin considerar desbalanceo).
    'micro' cuenta los verdaderos positivos globales.
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(f1_score(y_true, y_pred, average=average, zero_division=0))

def precision_multilabel(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'micro') -> float:
    """
    Calcula Precision para problemas multilabel.
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(precision_score(y_true, y_pred, average=average, zero_division=0))

def recall_multilabel(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'micro') -> float:
    """
    Calcula Recall para problemas multilabel.
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(recall_score(y_true, y_pred, average=average, zero_division=0))