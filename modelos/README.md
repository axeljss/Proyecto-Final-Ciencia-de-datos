# Carpeta `modelos/`

Un notebook por modelo, más un archivo con el código que comparten todos.
Esta carpeta va **dentro** de `Proyecto_Bank_Marketing/`, al mismo nivel que `data/`, `figuras/` y `resultados/`.

```
Proyecto_Bank_Marketing/
├── Proyecto_Final_Ciencia_Datos.ipynb   # Parte 1: datos y preparación (ya lo tienes)
├── data/
├── figuras/                             # las figuras de todos los notebooks caen aquí
├── resultados/                          # las tablas y métricas de todos los notebooks caen aquí
├── modelos/
│   ├── utils_proyecto.py                # código compartido (datos, validación cruzada, evaluación)
│   ├── 01_regresion_logistica.ipynb     # modelo de referencia (pivote)
│   ├── 02_naive_bayes.ipynb             # (pendiente)
│   ├── 03_svm.ipynb                     # (pendiente)
│   ├── 04_red_neuronal.ipynb            # (pendiente)
│   ├── 05_comparacion_final.ipynb       # (pendiente) tabla comparativa y análisis de errores
│   └── README.md
└── ...
```

## Cómo usarla

1. Copia la carpeta `modelos/` dentro de `Proyecto_Bank_Marketing/`.
2. En Anaconda Navigator abre **Jupyter Notebook**, entra a `modelos/` y abre `01_regresion_logistica.ipynb`.
3. Ejecuta con **Kernel → Restart & Run All**. Tarda menos de un minuto.

Importante: abre el notebook **desde dentro de `modelos/`**, porque ahí está `utils_proyecto.py`, que cada notebook importa con `import utils_proyecto as up`. Si aparece `ModuleNotFoundError: utils_proyecto`, el notebook no se abrió desde esa carpeta.

## Por qué hay un archivo común

`utils_proyecto.py` reproduce exactamente la limpieza y la división de la Parte 1 (misma semilla 42, misma división 70/15/15 estratificada) y define las funciones de evaluación. Así todos los modelos usan los mismos conjuntos y las mismas métricas, y la comparación es justa. Si cambias una regla (por ejemplo, el número de folds), la cambias en un solo lugar.

La primera celda de cada notebook imprime una tabla de control que debe dar **31,647 / 6,782 / 6,782 filas y 11.70 % de `yes`** en cada conjunto, igual que la sección 6 de la Parte 1.

## Cómo se comunican los notebooks entre sí

Cada notebook guarda en `resultados/`:

| Archivo | Contenido |
|---|---|
| `<modelo>_metricas_test.json` | Métricas en el conjunto de prueba y umbral usado |
| `<modelo>_predicciones_test.csv` | Probabilidad y predicción de cada cliente de *test* (análisis de errores) |
| `<modelo>_cv_scores.csv` | Puntajes por fold de la validación cruzada repetida (comparar modelos por pares) |
| `tabla_<modelo>_*.csv` | Tablas para el reporte |

El notebook de comparación final solo lee esos archivos, así que no hace falta volver a entrenar nada.
