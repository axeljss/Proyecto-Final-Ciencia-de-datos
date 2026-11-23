# Proyecto final de Ciencia de Datos

**Clasificación supervisada de clientes bancarios (Bank Marketing) con Naive Bayes, SVM y red neuronal**

**Autor(es):** _(nombre y correo)_
**Materia:** Ciencia de Datos. **Profesor:** M. en C. Benito Samuel López Razo

Se predice si un cliente contratará un **depósito a plazo** tras una campaña de llamadas telefónicas de un banco portugués, con el dataset *Bank Marketing* de UCI (45,211 registros, 16 variables). Se compararán tres modelos (Naive Bayes, máquina de soporte vectorial y perceptrón multicapa) contra una regresión logística de referencia.

> **Estado actual:** está lista la **Parte 1 (datos y preparación)**: EDA, limpieza, división train/validation/test y preprocesamiento. Los modelos se agregarán en el mismo notebook.

## Estructura

```
Proyecto_Bank_Marketing/
├── Proyecto_Final_Ciencia_Datos.ipynb   # Notebook con todo el análisis
├── data/
│   ├── bank-full.csv                    # Dataset del proyecto (45,211 registros)
│   ├── bank.csv                         # Muestra del 10 % de bank-full.csv (no se usa)
│   └── bank-names.txt                   # Descripción oficial del dataset
├── figuras/                             # Se generan al ejecutar el notebook
├── resultados/                          # Tablas en CSV y metadata.json (se generan al ejecutar)
├── reporte/                             # Artículo en Word y PDF (formato IEEE)
├── requirements.txt
└── README.md
```

## Instrucciones de ejecución (Anaconda + Jupyter Notebook)

### Opción A: Anaconda Navigator (la más sencilla)

1. Descomprime el zip en una carpeta, por ejemplo `C:\Users\TU_USUARIO\Proyecto_Bank_Marketing`.
2. Abre **Anaconda Navigator** y lanza **Jupyter Notebook**.
3. En la página que se abre en el navegador, entra a la carpeta del proyecto y abre `Proyecto_Final_Ciencia_Datos.ipynb`.
4. Ejecuta todo con **Kernel → Restart & Run All**.

### Opción B: Anaconda Prompt (entorno propio)

```
cd C:\Users\TU_USUARIO\Proyecto_Bank_Marketing
conda create -n ciencia_datos python=3.12 pandas numpy scikit-learn scipy matplotlib seaborn notebook -y
conda activate ciencia_datos
jupyter notebook
```

Luego abre `Proyecto_Final_Ciencia_Datos.ipynb` y usa **Kernel → Restart & Run All**.

### Verificar versiones

En una celda del notebook:

```python
import pandas, sklearn, seaborn, matplotlib
print(pandas.__version__, sklearn.__version__, seaborn.__version__, matplotlib.__version__)
```

Se requiere scikit-learn ≥ 1.2 y seaborn ≥ 0.13. Si alguna es más vieja, en Anaconda Prompt: `conda update scikit-learn seaborn` (o `pip install -U scikit-learn seaborn`) y reinicia el kernel.

> El notebook lee `data/bank-full.csv` con una ruta relativa, así que **debe abrirse desde la carpeta del proyecto** (donde está la carpeta `data/`). Con los pasos de arriba esto se cumple automáticamente.

## Reproducibilidad

- **Semilla fija:** `SEED = 42` para `numpy`, las divisiones de datos y (en las siguientes partes) los folds de validación cruzada y los modelos.
- **Sin fuga de información:** el preprocesamiento (escalado y codificación) se ajusta solo con el conjunto de entrenamiento; en la validación cruzada se usará dentro de un `Pipeline`.
- **Versiones:** `requirements.txt` indica las versiones mínimas; antes de entregar se fijan las exactas con `pip freeze`.

## Qué datos se usan

| Archivo | Uso |
|---|---|
| `bank-full.csv` | **Dataset del proyecto.** Todo el análisis y los modelos usan este archivo. |
| `bank.csv` | No se usa. Es una muestra aleatoria del 10 % de `bank-full.csv`; todas sus filas ya están en el archivo completo, así que mezclarlos duplicaría datos. |
| `bank-names.txt` | Solo documentación y cita del dataset. |

## Metodología resumida (Parte 1)

| Paso | Detalle |
|---|---|
| Datos | `bank-full.csv`, 45,211 registros, 16 variables + objetivo `y` |
| Limpieza | Sin nulos ni duplicados; `pdays = -1` convertido en indicador `pdays_contactado`; `unknown` conservado como categoría |
| Fuga de información | Se excluye `duration` (solo se conoce después de la llamada) |
| División | 70 % entrenamiento, 15 % validación, 15 % prueba (estratificada, semilla 42) |
| Preprocesamiento | Numéricas con `StandardScaler`; categóricas con *one-hot*; ajustado solo con *train* |

## Fuente de los datos

S. Moro, R. Laureano y P. Cortez, "Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology," en *Proc. European Simulation and Modelling Conference (ESM'2011)*, Guimarães, Portugal, oct. 2011, pp. 117–121. Dataset: UCI Machine Learning Repository, *Bank Marketing*. https://archive.ics.uci.edu/dataset/222/bank-marketing
