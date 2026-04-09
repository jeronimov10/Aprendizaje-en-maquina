import json, random

def uid():
    return ''.join(random.choices('0123456789abcdef', k=8))

def md(src):
    return {"cell_type": "markdown", "id": uid(), "metadata": {}, "source": src}

def code(src):
    return {"cell_type": "code", "execution_count": None, "id": uid(), "metadata": {}, "outputs": [], "source": src}

cells = [

md(["# Parte 1 — Iteración 7: Sin Strip de Acentos y Rango de N-grams Extendido\n",
    "\n",
    "En la iteración anterior el mejor modelo fue `char TF-IDF (2,6)` con LinearSVC, entrenado sobre el corpus completo. El score en el leaderboard fue 0.29.\n",
    "\n",
    "La hipótesis de esta iteración es que `strip_accents='unicode'` está descartando información temporal valiosa: el uso de ñ, á, é y otros diacríticos evolucionó considerablemente entre los siglos XVI y XIX. Mantener esos caracteres (`strip_accents=None`) debería mejorar la discriminación entre décadas.\n",
    "\n",
    "También exploramos extender el rango de n-grams para incluir caracteres individuales (1-grams) o secuencias más largas (7-grams)."]),

md(["## 1. Importación de librerías"]),

code(["import time\n",
      "import joblib\n",
      "import numpy as np\n",
      "import pandas as pd\n",
      "import matplotlib.pyplot as plt\n",
      "\n",
      "from sklearn.base import clone\n",
      "from sklearn.feature_extraction.text import TfidfVectorizer\n",
      "from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay\n",
      "from sklearn.model_selection import train_test_split\n",
      "from sklearn.pipeline import Pipeline\n",
      "from sklearn.svm import LinearSVC"]),

md(["## 2. Carga de los datos"]),

code(["TRAIN_PATH = 'train.csv'\n",
      "EVAL_PATH  = 'eval.csv'\n",
      "MODEL_PATH = 'best_model_v7.joblib'\n",
      "SUBMISSION_PATH = 'submission_v7.csv'"]),

code(["train_df = pd.read_csv(TRAIN_PATH)\n",
      "eval_df  = pd.read_csv(EVAL_PATH)\n",
      "\n",
      "train_df['text'] = train_df['text'].fillna('')\n",
      "eval_df['text']  = eval_df['text'].fillna('')\n",
      "\n",
      "print('Train shape:', train_df.shape)\n",
      "print('Eval shape: ', eval_df.shape)"]),

code(["train_df.head(3)"]),

md(["## 3. Exploración del conjunto de datos\n",
    "\n",
    "Revisión rápida de la distribución de clases y estadísticas básicas del texto."]),

code(["print('Décadas únicas:', sorted(train_df['decade'].unique()))\n",
      "print('Clases totales:', train_df['decade'].nunique())\n",
      "print()\n",
      "print('Distribución (primeras y últimas):')\n",
      "print(train_df['decade'].value_counts().sort_index())"]),

code(["train_df['n_chars'] = train_df['text'].str.len()\n",
      "train_df['n_words'] = train_df['text'].str.split().str.len()\n",
      "\n",
      "print(f\"Media de caracteres: {train_df['n_chars'].mean():.1f}\")\n",
      "print(f\"Mediana de caracteres: {train_df['n_chars'].median():.1f}\")\n",
      "print(f\"Media de palabras: {train_df['n_words'].mean():.1f}\")"]),

md(["## 4. Partición de los datos\n",
    "\n",
    "Usamos el mismo split que en iteraciones anteriores para que los resultados sean comparables."]),

code(["X = train_df['text']\n",
      "y = train_df['decade']\n",
      "\n",
      "X_train, X_val, y_train, y_val = train_test_split(\n",
      "    X, y, test_size=0.2, random_state=1, stratify=y\n",
      ")\n",
      "\n",
      "print('Train:', X_train.shape[0], '| Val:', X_val.shape[0])"]),

md(["## 5. Diseño de experimentos\n",
    "\n",
    "Definimos una función constructora de pipelines y un diccionario de candidatos. Cada candidato varía en:\n",
    "\n",
    "- `strip_accents`: `'unicode'` (iter6) vs `None` (hipótesis principal)\n",
    "- `ngram_range`: distintos rangos que incluyen 1-grams, 7-grams o ambos\n",
    "- `max_features`: vocabulario más grande a medida que el rango crece\n",
    "- `C`: valores de regularización para el LinearSVC"]),

code(["def build_char_model(c_value=0.75, ngram_range=(2, 6), max_features=180000,\n",
      "                     strip_accents=None, min_df=2):\n",
      "    return Pipeline([\n",
      "        ('tfidf', TfidfVectorizer(\n",
      "            lowercase=True,\n",
      "            strip_accents=strip_accents,\n",
      "            analyzer='char',\n",
      "            ngram_range=ngram_range,\n",
      "            min_df=min_df,\n",
      "            sublinear_tf=True,\n",
      "            max_features=max_features,\n",
      "            dtype=np.float32,\n",
      "        )),\n",
      "        ('clf', LinearSVC(C=c_value)),\n",
      "    ])"]),

code(["candidate_models = {\n",
      "    'baseline_iter6':    build_char_model(c_value=0.75, ngram_range=(2, 6), max_features=180000, strip_accents='unicode'),\n",
      "    'no_strip_v1':        build_char_model(c_value=0.75, ngram_range=(2, 6), max_features=180000, strip_accents=None),\n",
      "    'no_strip_v2_1_6':    build_char_model(c_value=0.75, ngram_range=(1, 6), max_features=200000, strip_accents=None),\n",
      "    'no_strip_v3_2_7':    build_char_model(c_value=0.75, ngram_range=(2, 7), max_features=200000, strip_accents=None),\n",
      "    'no_strip_v4_1_7':    build_char_model(c_value=0.75, ngram_range=(1, 7), max_features=250000, strip_accents=None),\n",
      "}"]),

md(["## 6. Entrenamiento y comparación de candidatos\n",
    "\n",
    "Entrenamos cada candidato en el conjunto de entrenamiento y evaluamos en validación."]),

code(["results = []\n",
      "\n",
      "for name, model in candidate_models.items():\n",
      "    t0 = time.time()\n",
      "    model.fit(X_train, y_train)\n",
      "    elapsed = time.time() - t0\n",
      "\n",
      "    preds = model.predict(X_val)\n",
      "    acc   = accuracy_score(y_val, preds)\n",
      "    f1    = f1_score(y_val, preds, average='macro')\n",
      "\n",
      "    results.append({'modelo': name, 'val_accuracy': acc, 'macro_f1': f1, 'tiempo_s': round(elapsed, 1)})\n",
      "    print(f'{name:<28} acc={acc:.4f}  f1={f1:.4f}  ({elapsed:.1f}s)')"]),

code(["results_df = pd.DataFrame(results).sort_values('val_accuracy', ascending=False).reset_index(drop=True)\n",
      "print(results_df.to_string(index=False))"]),

md(["Con el mejor rango y sin strip de acentos identificado, afinamos el valor de `C`."]),

code(["best_row      = results_df.iloc[0]\n",
      "best_name     = best_row['modelo']\n",
      "best_pipeline = candidate_models[best_name]\n",
      "best_ngram    = best_pipeline.named_steps['tfidf'].ngram_range\n",
      "best_maxf     = best_pipeline.named_steps['tfidf'].max_features\n",
      "\n",
      "print(f'Mejor candidato base: {best_name}')\n",
      "print(f'ngram_range={best_ngram}, max_features={best_maxf}')"]),

code(["c_candidates = {}\n",
      "for c_val in [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:\n",
      "    name  = f'c_tuning_{int(c_val*100):03d}'\n",
      "    model = build_char_model(c_value=c_val, ngram_range=best_ngram,\n",
      "                              max_features=best_maxf, strip_accents=None)\n",
      "    t0 = time.time()\n",
      "    model.fit(X_train, y_train)\n",
      "    elapsed = time.time() - t0\n",
      "\n",
      "    preds = model.predict(X_val)\n",
      "    acc   = accuracy_score(y_val, preds)\n",
      "    f1    = f1_score(y_val, preds, average='macro')\n",
      "\n",
      "    c_candidates[name] = {'model': model, 'val_accuracy': acc, 'macro_f1': f1, 'C': c_val}\n",
      "    print(f'{name}  C={c_val:.2f}  acc={acc:.4f}  f1={f1:.4f}  ({elapsed:.1f}s)')"]),

code(["tuning_df = pd.DataFrame([\n",
      "    {'C': v['C'], 'val_accuracy': v['val_accuracy'], 'macro_f1': v['macro_f1']}\n",
      "    for v in c_candidates.values()\n",
      "]).sort_values('val_accuracy', ascending=False)\n",
      "\n",
      "print(tuning_df.to_string(index=False))"]),

md(["## 7. Evaluación del mejor modelo\n",
    "\n",
    "Seleccionamos la mejor combinación final y revisamos el reporte de clasificación completo sobre el conjunto de validación."]),

code(["best_c_name  = max(c_candidates, key=lambda k: c_candidates[k]['val_accuracy'])\n",
      "best_c_entry = c_candidates[best_c_name]\n",
      "best_model   = best_c_entry['model']\n",
      "\n",
      "print(f'Mejor modelo: {best_c_name}')\n",
      "print(f\"Val accuracy: {best_c_entry['val_accuracy']:.4f}\")\n",
      "print(f\"Macro F1:     {best_c_entry['macro_f1']:.4f}\")"]),

code(["y_val_pred = best_model.predict(X_val)\n",
      "print(classification_report(y_val, y_val_pred))"]),

code(["fig, ax = plt.subplots(figsize=(16, 12))\n",
      "ConfusionMatrixDisplay.from_predictions(y_val, y_val_pred, ax=ax, colorbar=False)\n",
      "plt.title('Matriz de confusión — validación (iter7)')\n",
      "plt.tight_layout()\n",
      "plt.show()"]),

md(["## 8. Predicciones sobre eval.csv\n",
    "\n",
    "Reentrenamos el mejor pipeline sobre el conjunto de entrenamiento completo (train + validación) antes de generar las predicciones finales."]),

code(["final_model = clone(best_model)\n",
      "final_model.fit(X, y)\n",
      "\n",
      "joblib.dump(final_model, MODEL_PATH)\n",
      "print(f'Modelo guardado en: {MODEL_PATH}')"]),

code(["eval_predictions = final_model.predict(eval_df['text']).astype(int)\n",
      "\n",
      "submission_df = pd.DataFrame({\n",
      "    'id':     eval_df['id'],\n",
      "    'answer': eval_predictions,\n",
      "})\n",
      "submission_df.to_csv(SUBMISSION_PATH, index=False)\n",
      "\n",
      "print(f'Submission guardado en: {SUBMISSION_PATH}')\n",
      "submission_df.head()"]),

]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

path = 'c:/Users/jeron/OneDrive/Escritorio/Programacion/Aprendizaje-en-maquina/Proyectos/parte1_iteracion7.ipynb'
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f'Escrito: {path}')
