"""Código compartido por los notebooks de modelos del proyecto Bank Marketing.

Este archivo vive en la carpeta modelos/ y se importa desde cada notebook con:
    import utils_proyecto as up

Reproduce EXACTAMENTE la preparación de la Parte 1 (misma limpieza, misma semilla y
la misma división 70/15/15 estratificada), así que todos los modelos trabajan con
los mismos datos y los resultados son comparables.
"""
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuración (la misma de la Parte 1)
# ---------------------------------------------------------------------------
SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
EXCLUIR_DURATION = True

# Rutas: se calculan a partir de la ubicación de este archivo (modelos/), así funcionan
# sin importar desde qué carpeta se ejecute el notebook.
RAIZ = Path(__file__).resolve().parent.parent
DATA = RAIZ / "data" / "bank-full.csv"
FIGURAS = RAIZ / "figuras"
RESULTADOS = RAIZ / "resultados"
for _carpeta in (FIGURAS, RESULTADOS):
    _carpeta.mkdir(parents=True, exist_ok=True)

# Validación cruzada para ajustar hiperparámetros (se hace DENTRO de train)
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Validación cruzada repetida para medir la variabilidad (mismas particiones para todos los modelos)
CV_REP = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)

# Métrica con la que se elige la mejor combinación de hiperparámetros
METRICA_CV = "average_precision"

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight"})


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
@dataclass
class Datos:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    preprocesador: ColumnTransformer   # SIN ajustar: úsalo con clone() dentro de un Pipeline
    cols_num: list
    cols_cat: list
    cols_bin: list


def construir_preprocesador(cols_num, cols_cat, cols_bin, yeo_johnson=False):
    # Numéricas: estandarizar (o Yeo-Johnson + estandarizar); categóricas: one-hot; binaria: se deja igual.
    escalador = PowerTransformer(method="yeo-johnson", standardize=True) if yeo_johnson else StandardScaler()
    prep = ColumnTransformer(
        transformers=[
            ("num", escalador, cols_num),
            ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False), cols_cat),
            ("bin", "passthrough", cols_bin),
        ],
        verbose_feature_names_out=False,
    )
    prep.set_output(transform="pandas")
    return prep


def cargar_datos():
    # Limpieza y división idénticas a las secciones 5 y 6 de la Parte 1.
    if not DATA.exists():
        raise FileNotFoundError(f"No encontré {DATA}. Verifica que bank-full.csv esté en la carpeta data/ del proyecto.")
    df = pd.read_csv(DATA, sep=";")
    for col in df.select_dtypes(exclude="number").columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

    df["y"] = df["y"].map({"no": 0, "yes": 1}).astype(int)
    df = df.drop_duplicates().reset_index(drop=True)
    df["pdays_contactado"] = (df["pdays"] != -1).astype(int)
    df.loc[df["pdays"] == -1, "pdays"] = 0
    if EXCLUIR_DURATION:
        df = df.drop(columns=["duration"])

    X = df.drop(columns="y")
    y = df["y"]
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE / (1 - TEST_SIZE), stratify=y_temp, random_state=SEED
    )

    cols_bin = ["pdays_contactado"]
    cols_cat = list(X_train.select_dtypes(exclude="number").columns)
    cols_num = [c for c in X_train.select_dtypes(include="number").columns if c not in cols_bin]
    prep = construir_preprocesador(cols_num, cols_cat, cols_bin, yeo_johnson=False)
    return Datos(X_train, X_val, X_test, y_train, y_val, y_test, prep, cols_num, cols_cat, cols_bin)


def resumen_datos(d):
    # Tabla de control: debe coincidir con la sección 6 de la Parte 1.
    return pd.DataFrame({
        "filas": [len(d.X_train), len(d.X_val), len(d.X_test)],
        "% clase 'yes'": [round(s.mean() * 100, 2) for s in (d.y_train, d.y_val, d.y_test)],
    }, index=["train", "validation", "test"])


# ---------------------------------------------------------------------------
# Guardado
# ---------------------------------------------------------------------------
def guardar(fig, nombre):
    ruta = FIGURAS / f"{nombre}.png"
    fig.savefig(ruta)
    print(f"Figura guardada: figuras/{ruta.name}")


def guardar_tabla(tabla, nombre):
    ruta = RESULTADOS / f"{nombre}.csv"
    tabla.to_csv(ruta, encoding="utf-8-sig")
    print(f"Tabla guardada: resultados/{ruta.name}")


def formatear_params(params):
    # {'modelo__C': 0.1, 'modelo__class_weight': None} -> 'C = 0.1, class_weight = None'
    partes = []
    for clave, valor in params.items():
        nombre = clave.split("__")[-1]
        partes.append(f"{nombre} = {valor:g}" if isinstance(valor, (int, float, np.number)) else f"{nombre} = {valor}")
    return ", ".join(partes)


# ---------------------------------------------------------------------------
# Evaluación (la misma para todos los modelos)
# ---------------------------------------------------------------------------
def umbral_optimo_f1(y_true, proba):
    # Umbral de decisión que maximiza F1 (se calcula SOLO con validación).
    precision, recall, umbrales = precision_recall_curve(y_true, proba)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(umbrales[np.argmax(f1)])


def evaluar_modelo(nombre, modelo, slug, d):
    # 1) El umbral se elige con validación
    proba_val = modelo.predict_proba(d.X_val)[:, 1]
    umbral = umbral_optimo_f1(d.y_val, proba_val)

    # 2) Evaluación única en test
    proba_test = modelo.predict_proba(d.X_test)[:, 1]
    pred_test = (proba_test >= umbral).astype(int)
    y_test = d.y_test

    metricas = {
        "umbral": umbral,
        "exactitud": accuracy_score(y_test, pred_test),
        "precisión": precision_score(y_test, pred_test),
        "sensibilidad": recall_score(y_test, pred_test),
        "F1": f1_score(y_test, pred_test),
        "AUC-ROC": roc_auc_score(y_test, proba_test),
        "AUC-PR": average_precision_score(y_test, proba_test),
    }

    # 3) Se guardan en disco para la comparación final y el análisis de errores
    (RESULTADOS / f"{slug}_metricas_test.json").write_text(
        json.dumps({"modelo": nombre, **{k: float(v) for k, v in metricas.items()}}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    pd.DataFrame({"y_real": y_test, "proba": proba_test, "pred": pred_test}, index=y_test.index).to_csv(
        RESULTADOS / f"{slug}_predicciones_test.csv", index_label="id_original"
    )

    # 4) Matriz de confusión, curva ROC y curva precisión-recall
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred_test, display_labels=["no", "yes"], cmap="Blues", colorbar=False, ax=axes[0]
    )
    axes[0].set_title(f"Matriz de confusión (umbral = {umbral:.2f})")
    axes[0].set_xlabel("Predicción")
    axes[0].set_ylabel("Real")
    axes[0].grid(False)

    fpr, tpr, _ = roc_curve(y_test, proba_test)
    axes[1].plot(fpr, tpr, label=f"AUC-ROC = {metricas['AUC-ROC']:.3f}")
    axes[1].plot([0, 1], [0, 1], "--", color="gray", label="azar")
    axes[1].set_xlabel("Tasa de falsos positivos")
    axes[1].set_ylabel("Tasa de verdaderos positivos")
    axes[1].set_title("Curva ROC")
    axes[1].legend(loc="lower right")

    prec, rec, _ = precision_recall_curve(y_test, proba_test)
    axes[2].plot(rec, prec, label=f"AUC-PR = {metricas['AUC-PR']:.3f}")
    axes[2].axhline(y_test.mean(), ls="--", color="gray", label=f"azar = {y_test.mean():.3f}")
    axes[2].set_xlabel("Sensibilidad (recall)")
    axes[2].set_ylabel("Precisión")
    axes[2].set_title("Curva precisión-recall")
    axes[2].legend(loc="upper right")

    fig.suptitle(nombre, y=1.03)
    fig.tight_layout()
    plt.show()
    guardar(fig, f"fig_{slug}_evaluacion")
    print(f"Métricas guardadas: resultados/{slug}_metricas_test.json y resultados/{slug}_predicciones_test.csv")
    return pd.Series(metricas, name=nombre).round(4)


def variabilidad_cv(nombre, modelo, slug, d, n_jobs=-1):
    # Media y desviación estándar entre folds (validación cruzada repetida 5 x 3 sobre train)
    scoring = {"AUC-PR": "average_precision", "AUC-ROC": "roc_auc"}
    res = cross_validate(clone(modelo), d.X_train, d.y_train, cv=CV_REP, scoring=scoring, n_jobs=n_jobs)
    por_fold = pd.DataFrame({m: res[f"test_{m}"] for m in scoring})
    por_fold.to_csv(RESULTADOS / f"{slug}_cv_scores.csv", index_label="fold", encoding="utf-8-sig")
    tabla = pd.DataFrame({"media": por_fold.mean(), "desv. estándar": por_fold.std(ddof=0)}).round(4)
    guardar_tabla(tabla, f"tabla_{slug}_variabilidad_cv")
    return tabla


def cramers_v(x, y):
    # V de Cramér entre dos variables categóricas (0 = sin asociación, 1 = asociación perfecta)
    tabla = pd.crosstab(x, y)
    chi2, p, _, _ = chi2_contingency(tabla)
    n = tabla.values.sum()
    k = min(tabla.shape) - 1
    return float(np.sqrt(chi2 / (n * k))), p
