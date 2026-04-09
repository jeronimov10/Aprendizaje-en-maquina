"""Genera P_5JVP.ipynb — Vocabulario Óptimo con char n-grams + LinearSVC + GridSearchCV."""
import json, uuid

def cell_id():
    return uuid.uuid4().hex[:8]

def code(lines):
    return {"cell_type": "code", "id": cell_id(), "metadata": {},
            "source": lines, "outputs": [], "execution_count": None}

def md(lines):
    return {"cell_type": "markdown", "id": cell_id(), "metadata": {},
            "source": lines}

cells = [

# ── TÍTULO ──────────────────────────────────────────────────────────────────
md(["# Vocabulario Óptimo: max_features y min_df con Character N-Grams + LinearSVC\n",
    "\n",
    "Este notebook profundiza en los hallazgos de la iteración 7 (Kaggle 0.30085) explorando de forma sistemática la combinación de **tamaño de vocabulario** (`max_features`) y **frecuencia mínima de documento** (`min_df`) mediante `GridSearchCV` con validación cruzada estratificada.\n",
    "\n",
    "Se utilizan dos rejillas de hiperparámetros separadas para respetar las restricciones de memoria:\n",
    "- **Rejilla A — `min_df=1`**: incluye n-grams que aparecen una sola vez; requiere vocabularios más pequeños (≤ 200 k) para no desbordar la RAM.\n",
    "- **Rejilla B — `min_df=2`**: filtra hápax legómena; permite vocabularios más grandes (≤ 350 k).\n",
    "\n",
    "El clasificador es `LinearSVC`, el de mayor rendimiento para TF-IDF de alta dimensionalidad. "
    "El pipeline garantiza que la transformación TF-IDF se ajuste únicamente sobre los datos de entrenamiento de cada fold."]),

# ── 1. IMPORTS ───────────────────────────────────────────────────────────────
md(["## 1. Importación de librerías"]),

code(["import numpy as np\n",
      "import pandas as pd\n",
      "import matplotlib.pyplot as plt\n",
      "\n",
      "from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score\n",
      "from sklearn.pipeline import Pipeline\n",
      "from sklearn.feature_extraction.text import TfidfVectorizer\n",
      "from sklearn.svm import LinearSVC\n",
      "from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay"]),

# ── 2. CARGA ─────────────────────────────────────────────────────────────────
md(["## 2. Carga de los datos\n",
    "\n",
    "Cargamos el conjunto de entrenamiento y el de evaluación de forma independiente."]),

code(["df = pd.read_csv('train.csv')\n",
      "df_eval = pd.read_csv('eval.csv')\n",
      "\n",
      "df['text']      = df['text'].fillna('')\n",
      "df_eval['text'] = df_eval['text'].fillna('')"]),

code(["df.head()"]),

code(["df.shape"]),

# ── 3. EXPLORACIÓN ──────────────────────────────────────────────────────────
md(["## 3. Exploración del conjunto de datos\n",
    "\n",
    "Revisamos la distribución de la variable objetivo y la calidad general del conjunto."]),

code(["df['decade'].value_counts().sort_index().plot(\n",
      "    kind='bar', figsize=(14, 4), title='Distribución de décadas'\n",
      ")\n",
      "plt.xlabel('Década')\n",
      "plt.ylabel('Frecuencia')\n",
      "plt.tight_layout()\n",
      "plt.show()"]),

code(["def reporte_calidad(df):\n",
      "    reporte_de_cualidad = {\n",
      "        'Total records': len(df),\n",
      "        'duplicated record': df.duplicated().sum(),\n",
      "        'missing values': df.isnull().sum().to_dict(),\n",
      "        'data types': df.dtypes.astype(str).to_dict(),\n",
      "    }\n",
      "    return reporte_de_cualidad\n",
      "\n",
      "print(reporte_calidad(df))"]),

# ── 4. PREPROCESAMIENTO ──────────────────────────────────────────────────────
md(["## 4. Preprocesamiento del texto\n",
    "\n",
    "Para textos históricos en español (siglos XVI–XIX) la ortografía **es** la señal temporal: la ñ, las tildes y las grafías arcaicas evolucionaron de forma diferente en cada época. "
    "Por eso desactivamos explícitamente `strip_accents=None` dentro del vectorizador y evitamos cualquier normalización que destruya esa información.\n",
    "\n",
    "La única limpieza que realizamos es eliminar duplicados exactos."]),

code(["df = df.drop_duplicates().reset_index(drop=True)\n",
      "\n",
      "print('Registros tras eliminar duplicados:', len(df))\n",
      "df[['text', 'decade']].head()"]),

# ── 5. PARTICIÓN ─────────────────────────────────────────────────────────────
md(["## 5. Partición de los datos"]),

code(["X = df['text']\n",
      "y = df['decade']\n",
      "\n",
      "X_train, X_val, y_train, y_val = train_test_split(\n",
      "    X, y, test_size=0.2, random_state=1, stratify=y\n",
      ")"]),

md(["Se usa `stratify=y` para mantener la proporción de clases en ambas particiones."]),

code(["X_train.shape, X_val.shape"]),

# ── 6. PIPELINE ──────────────────────────────────────────────────────────────
md(["## 6. Construcción del pipeline\n",
    "\n",
    "El pipeline encadena el vectorizador TF-IDF de caracteres con `LinearSVC`. "
    "Los hiperparámetros base se fijan en los valores óptimos ya validados en las iteraciones anteriores:\n",
    "- `analyzer='char'`: captura morfología y ortografía directamente.\n",
    "- `ngram_range=(1, 7)`: desde caracteres individuales hasta heptagramas.\n",
    "- `strip_accents=None`: preserva toda la información diacrítica.\n",
    "- `sublinear_tf=True`: escala logarítmica para reducir el peso de n-grams muy frecuentes.\n",
    "- `dtype=np.float32`: reduce el uso de memoria a la mitad respecto a float64.\n",
    "\n",
    "Los hiperparámetros `max_features`, `min_df` y el parámetro de regularización `C` se exploran mediante `GridSearchCV`."]),

code(["pipeline_vocab = Pipeline([\n",
      "    ('tfidf', TfidfVectorizer(\n",
      "        lowercase=True,\n",
      "        strip_accents=None,\n",
      "        analyzer='char',\n",
      "        ngram_range=(1, 7),\n",
      "        sublinear_tf=True,\n",
      "        dtype=np.float32,\n",
      "    )),\n",
      "    ('clf', LinearSVC()),\n",
      "])"]),

# ── 7. GRID SEARCH ───────────────────────────────────────────────────────────
md(["## 7. Búsqueda de hiperparámetros\n",
    "\n",
    "Se utilizan **dos rejillas separadas** para evitar el desbordamiento de memoria que ocurre cuando `min_df=1` se combina con vocabularios grandes (≥ 300 k features):\n",
    "\n",
    "| Rejilla | `min_df` | `max_features` | Motivo |\n",
    "|---------|----------|----------------|--------|\n",
    "| A | 1 | 150 k, 200 k | `min_df=1` produce vocabularios explosivos; se limita el tamaño |\n",
    "| B | 2 | 250 k, 300 k, 350 k | `min_df=2` filtra hápax; permite vocabularios más grandes |\n",
    "\n",
    "Ambas rejillas comparten el mismo rango de `C` = [0.25, 0.35, 0.40, 0.50]."]),

code(["param_grid_mindf1 = {\n",
      "    'tfidf__min_df':      [1],\n",
      "    'tfidf__max_features': [150000, 200000],\n",
      "    'clf__C':              [0.25, 0.35, 0.40, 0.50],\n",
      "}\n",
      "\n",
      "param_grid_mindf2 = {\n",
      "    'tfidf__min_df':      [2],\n",
      "    'tfidf__max_features': [250000, 300000, 350000],\n",
      "    'clf__C':              [0.25, 0.35, 0.40, 0.50],\n",
      "}\n",
      "\n",
      "param_grid = [param_grid_mindf1, param_grid_mindf2]"]),

code(["kfold = KFold(n_splits=3, shuffle=True, random_state=0)\n",
      "grid_vocab = GridSearchCV(\n",
      "    pipeline_vocab, param_grid, cv=kfold,\n",
      "    scoring='accuracy', n_jobs=-1, verbose=2\n",
      ")"]),

code(["grid_vocab.fit(X_train, y_train)"]),

code(["print('Mejores hiperparámetros:', grid_vocab.best_params_)\n",
      "print('Mejor score CV (accuracy):', round(grid_vocab.best_score_, 4))"]),

code(["resultados_cv = pd.DataFrame(grid_vocab.cv_results_)\n",
      "cols = ['param_clf__C', 'param_tfidf__min_df', 'param_tfidf__max_features',\n",
      "        'mean_test_score', 'std_test_score', 'rank_test_score']\n",
      "print(resultados_cv[cols].sort_values('rank_test_score').to_string(index=False))"]),

# ── 8. EVALUACIÓN ────────────────────────────────────────────────────────────
md(["## 8. Evaluación del mejor modelo\n",
    "\n",
    "Aplicamos el mejor pipeline encontrado por `GridSearchCV` sobre el conjunto de validación."]),

code(["best_model = grid_vocab.best_estimator_\n",
      "\n",
      "y_pred_train = best_model.predict(X_train)\n",
      "y_pred_val   = best_model.predict(X_val)"]),

md(["#### Comparación de rendimientos sobre entrenamiento y validación"]),

code(["print('Accuracy en entrenamiento:', round(accuracy_score(y_train, y_pred_train), 4))\n",
      "print('Accuracy en validación:   ', round(accuracy_score(y_val,   y_pred_val),   4))\n",
      "print('Mejor score CV (accuracy):', round(grid_vocab.best_score_,                4))"]),

code(["print(classification_report(y_val, y_pred_val))"]),

code(["fig, ax = plt.subplots(figsize=(16, 12))\n",
      "ConfusionMatrixDisplay.from_predictions(y_val, y_pred_val, ax=ax, colorbar=False)\n",
      "plt.title('Matriz de confusión — validación')\n",
      "plt.tight_layout()\n",
      "plt.show()"]),

# ── 9. PREDICCIONES ──────────────────────────────────────────────────────────
md(["## 9. Predicciones sobre eval.csv\n",
    "\n",
    "Reentrenamos el mejor pipeline sobre el conjunto de entrenamiento completo antes de generar las predicciones finales. "
    "Esto aprovecha todos los datos disponibles y produce un modelo más robusto que el evaluado en el paso anterior."]),

code(["from sklearn.base import clone\n",
      "\n",
      "final_model = clone(best_model)\n",
      "final_model.fit(X, y)\n",
      "\n",
      "print('Modelo reentrenado sobre el conjunto completo.')"]),

code(["y_eval_pred = final_model.predict(df_eval['text']).astype(int)\n",
      "\n",
      "submission = pd.DataFrame({'id': df_eval['id'], 'answer': y_eval_pred})\n",
      "submission.to_csv('submission_vocab_opt.csv', index=False)\n",
      "\n",
      "print('Filas generadas:', len(submission))\n",
      "submission.head()"]),

]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

out = "c:/Users/jeron/OneDrive/Escritorio/Programacion/Aprendizaje-en-maquina/Proyectos/P_5JVP.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Escrito:", out)
