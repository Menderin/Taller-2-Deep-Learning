# Laboratorio 02: Redes Neuronales Poco Profundas

**Modelos en PyTorch para la Predicción del Deterioro Cognitivo**
*Deep Learning — Universidad Católica del Norte*
*Prof. Dr. Juan Bekios Calfa*

---

## Descripción general

Este repositorio contiene la implementación del **Laboratorio 02**, cuyo objetivo es desarrollar y evaluar redes neuronales poco profundas (*Shallow Networks*) en PyTorch para estudiar el deterioro cognitivo mediante **seis experimentos independientes**, siguiendo la metodología **CRISP-DM**.

Cada experimento corresponde a una formulación distinta del problema (una columna objetivo GDS diferente), tratada como una tarea de **clasificación multilabel con codificación one-hot**. La pipeline completa incluye preprocesamiento, búsqueda de hiperparámetros, validación cruzada anidada y estimación de incertidumbre predictiva.

---

## Objetivos específicos

- Comprender el deterioro cognitivo como una colección de seis experimentos supervisados independientes.
- Analizar las variables objetivo `GDS`, `GDS_R1`, `GDS_R2`, `GDS_R3`, `GDS_R4` y `GDS_R5`, cada una con un número de clases y distribución distinta.
- Codificar cada columna objetivo como un vector multilabel tipo one-hot antes del entrenamiento.
- Implementar `ShallowCognitiveNet` en PyTorch, cuyo número de neuronas de salida se adapta al experimento activo.
- Aplicar **Validación Cruzada Anidada** (5 folds externos × 3 folds internos) con estratificación multilabel para evitar data leakage y optimismo artificial.
- Estimar incertidumbre predictiva mediante **Monte Carlo Dropout** (N=30 pasadas).

---

## Dataset

El dataset (`data/raw/15 atributos R0-R5.sav`) contiene **1119 observaciones** y los siguientes campos:

| Grupo | Atributos |
|---|---|
| Orientación Temporal | `Día`, `Mes`, `Año`, `Estación` |
| Orientación Espacial | `País`, `Ciudad`, `CalleLugar`, `NumeroPiso` |
| Memoria / Retención | `Miguel2`, `González2`, `Avenida2`, `Imperial2`, `A682`, `Caldera2`, `Copiapo2` |

Los 15 atributos de entrada son **completamente binarios**. El campo `ID` no se usa como variable predictiva.

**Variables objetivo (6 experimentos):**

| Target | Descripción |
|---|---|
| `GDS` | Escala original. Referencia metodológica (mal condicionada). |
| `GDS_R1` | Reagrupación 1 de clases GDS. |
| `GDS_R2` | Reagrupación 2 de clases GDS. |
| `GDS_R3` | Reagrupación 3 de clases GDS. |
| `GDS_R4` | Reagrupación 4 de clases GDS. |
| `GDS_R5` | Reagrupación 5 de clases GDS. |

Cada columna objetivo se convierte a formato one-hot: si un experimento tiene K clases y una muestra pertenece a la clase 2, su codificación es `[0, 1, 0, ..., 0]`.

---

## Arquitectura del modelo

`ShallowCognitiveNet` es una red neuronal poco profunda con la siguiente estructura:

```
Entrada (15) → Capa Oculta (hidden_dim) → Activación → Dropout → Salida (K clases)
```

- **Capa de entrada:** recibe los 15 atributos binarios.
- **Capa oculta:** aprende una representación intermedia compartida.
- **Activación:** configurable (`ReLU`, `Mish`, u otras definidas en `config.py`).
- **Dropout:** regularización y fuente de incertidumbre en inferencia.
- **Capa de salida:** produce un logit por cada clase del experimento activo.
- **Función de pérdida:** `BCEWithLogitsLoss` (combina sigmoide + entropía cruzada binaria por etiqueta; **no** se aplica sigmoide dentro del modelo).

---

## Estructura del repositorio

```
Taller-2-Deep-Learning/
├── data/
│   ├── raw/                  # Dataset original (.sav) — NO incluido en el repo
│   └── processed/            # CSV limpio (auto-generado por la pipeline)
├── docs/                     # Diapositivas y material teórico del laboratorio
├── results/                  # Resultados por activación y target
│   ├── Mish/
│   │   └── <Target>/
│   │       ├── best_model_fold_N.pth
│   │       ├── class_distribution.png
│   │       ├── scaling_comparison.png
│   │       ├── uncertainty_distribution.png
│   │       └── uncertainty_vs_probability.png
│   ├── ReLU/
│   │   └── <Target>/
│   │       ├── best_model_fold_N.pth
│   │       ├── class_distribution.png
│   │       ├── scaling_comparison.png
│   │       ├── uncertainty_distribution.png
│   │       └── uncertainty_vs_probability.png
│   ├── comparison_summary.csv
│   ├── comparison_summary.json
│   └── tabla_comparacion.md
├── src/
│   ├── core/
│   │   ├── config.py         # Hiperparámetros base, columnas y configuración
│   │   └── enviroment.yml    # Dependencias del entorno
│   ├── models/network.py     # ShallowCognitiveNet (PyTorch nn.Module)
│   ├── data_loader.py        # Carga de .sav/.csv y CognitiveMultiLabelDataset
│   ├── preprocessing.py      # Tensores, imputación, one-hot e índices de folds
│   ├── training.py           # Random Search distribuido y loop de entrenamiento
│   ├── evaluation.py         # Métricas multilabel (F1, Hamming, Exact Match, etc.)
│   ├── uncertainty.py        # Inferencia bayesiana vía Monte Carlo Dropout
│   └── visualization.py     # Generación automatizada de gráficos pedagógicos
├── exploration.py            # Script de minería y exploración del CSV
├── main.py                   # Orquestador principal — ejecutar desde aquí
└── README.md
```

---

## Flujo experimental (CRISP-DM)

La pipeline sigue estas etapas para cada uno de los 6 experimentos × cada función de activación:

1. **Comprensión del problema:** cada columna objetivo define un experimento distinto.
2. **Comprensión de los datos:** exploración de distribución de clases y atributos.
3. **Preparación de los datos:** codificación one-hot multilabel, imputación y escalado.
4. **Modelado con Nested Cross-Validation:**
   - **5 folds externos** → estimación del desempeño final (sin contaminación).
   - **3 folds internos** → selección de hiperparámetros por Random Search.
   - Reentrenamiento del mejor modelo sobre todo el fold externo.
5. **Evaluación:** métricas multilabel por fold externo + Monte Carlo Dropout.
6. **Generación de resultados:** gráficos, pesos `.pth` y tabla comparativa.

### Hiperparámetros explorados

| Hiperparámetro | Valores |
|---|---|
| `hidden_dim` | 16, 32, 64 |
| `dropout_prob` | 0.2, 0.3, 0.5 |
| `lr` | 1e-2, 1e-3 |
| `weight_decay` | 0, 1e-4 |
| `batch_size` | 16, 32 , 64|

---

## Métricas de evaluación

Se reportan las siguientes métricas multilabel (promediadas sobre los 5 folds externos):

| Métrica | Descripción |
|---|---|
| **Hamming Loss** | Proporción de etiquetas mal clasificadas (↓ mejor) |
| **Exact Match Ratio** | Fracción de muestras con el vector completo correcto (↑ mejor) |
| **Precision Micro** | Precisión global sumando TP/FP sobre todas las etiquetas |
| **Recall Micro** | Cobertura global sumando TP/FN sobre todas las etiquetas |
| **F1 Micro** | Media armónica de Precision y Recall Micro |
| **F1 Macro** | F1 por etiqueta promediado (sensible a clases minoritarias) |
| **MC Uncertainty** | Varianza promedio de las N=30 pasadas de Monte Carlo Dropout |

---

## Incertidumbre predictiva (Monte Carlo Dropout)

Durante la inferencia, el Dropout permanece **activo** (`model.train()`). Se realizan **30 pasadas hacia adelante** sobre cada batch y se calcula:

- **Media de probabilidades** por etiqueta → predicción final.
- **Varianza de probabilidades** por etiqueta → incertidumbre del modelo.

Una dispersión alta entre pasadas indica mayor duda del modelo sobre esa etiqueta. Esto es especialmente relevante en contextos clínicos, donde no todas las predicciones deben interpretarse con la misma confianza.

---

## Instalación y uso

### Requisitos

```bash
conda env create -f src/core/environment.yml
conda activate lab02-dl
```

### Preparar los datos

Coloca el archivo original en:

```
data/raw/15 atributos R0-R5.sav
```

Si el `.sav` no está disponible, la pipeline genera automáticamente un dataset binario dummy para propósitos de demostración.

### Ejecutar la pipeline completa

```bash
python main.py
```

El orquestador detecta automáticamente la GPU disponible (con soporte multi-GPU vía `DataParallel`) y usa `joblib` para paralelizar el Random Search sobre todos los hilos de CPU disponibles.

### Exploración de datos

```bash
python exploration.py
```

---

## Resultados generados

Al finalizar, el directorio `results/` contiene:

- **Gráficos por experimento:** distribución de clases, comparación de escalamiento, distribución de incertidumbre y scatter de incertidumbre vs. probabilidad.
- **Pesos del modelo:** `best_model_fold_N.pth` por cada fold externo y función de activación.
- **Tabla comparativa:** `results/tabla_comparacion.md` con F1 Macro, F1 Micro, Hamming Loss y Exact Match por target y activación.
- **Resumen CSV/JSON:** `comparison_summary.csv` y `comparison_summary.json` para análisis posterior.

---

## Rendimiento en HPC

- **Random Search paralelo (`joblib`):** utiliza el 100% de los hilos lógicos disponibles para la búsqueda de hiperparámetros en RAM.
- **DataLoaders asíncronos:** `pin_memory=True` y workers configurados a `cpu_count() // 2`.
- **Multi-GPU (`DataParallel`):** si se detectan múltiples GPUs, las redes se sincronizan automáticamente en paralelo.

---

## Criterios de evaluación

| Criterio | Ponderación |
|---|---|
| Comprensión del problema multilabel | 20% |
| Implementación de la red en PyTorch | 20% |
| Validación y diseño experimental | 25% |
| Interpretación de métricas e incertidumbre | 20% |
| Organización modular del código y reproducibilidad | 15% |

---

## Bibliografía

- Géron, A. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* — capítulos sobre redes neuronales y evaluación.
- Bishop, C. *Pattern Recognition and Machine Learning* — capítulos sobre clasificación probabilística.
- Goodfellow, Bengio y Courville. *Deep Learning* — secciones sobre redes feedforward y regularización.
- Documentación oficial de [PyTorch](https://pytorch.org/docs/) (`nn.Module`, `Dropout`, `BCEWithLogitsLoss`).
- Documentación oficial de [scikit-learn](https://scikit-learn.org/) — métricas multilabel y validación cruzada.

---

*Laboratorio desarrollado para el curso de Deep Learning — Universidad Católica del Norte, 2026.*
