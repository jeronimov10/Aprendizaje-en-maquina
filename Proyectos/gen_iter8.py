import json, random

def uid():
    return ''.join(random.choices('0123456789abcdef', k=8))

def md(src):
    return {"cell_type": "markdown", "id": uid(), "metadata": {}, "source": src}

def code(src):
    return {"cell_type": "code", "execution_count": None, "id": uid(), "metadata": {}, "outputs": [], "source": src}

cells = [

md(["# Parte 1 — Iteración 8: min_df=1 y Vocabulario Extendido\n",
    "\n",
    "En la iteración 7 se comprobó que `strip_accents=None` con `ngram_range=(1,7)` y `max_features=250k` (C=0.40) supera claramente la configuración anterior. El score en Kaggle fue 0.30085.\n",
    "\n",
    "Esta iteración explora dos hipótesis adicionales:\n",
    "\n",
    "1. **`min_df=1`**: actualmente se descartan n-grams que aparecen solo una vez. Para textos históricos con 39 clases finas, patrones ortográficos rarísimos pero específicos de una sola década podrían ser altamente discriminativos. Se limita `max_features` a 150k–200k para evitar desbordamiento de memoria.\n",
    "2. **Vocabulario más grande con `min_df=2`**: aumentar `max_features` a 300k–350k manteniendo `min_df=2` (seguro en memoria) para capturar más n-grams frecuentes."]),

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

code(["TRAIN_PATH      = 'train.csv'\n",
      "EVAL_PATH       = 'eval.csv'\n",
      "MODEL_PATH      = 'best_model_v8.joblib'\n",
      "SUBMISSION_PATH = 'submission_v8.csv'"]),

code(["train_df = pd.read_csv(TRAIN_PATH)\n",
      "eval_df  = pd.read_csv(EVAL_PATH)\n",
      "\n",
      "train_df['text'] = train_df['text'].fillna('')\n",
      "eval_df['text']  = eval_df['text'].fillna('')\n",
      "\n",
      "print('Train shape:', train_df.shape)\n",
      "print('Eval shape: ', eval_df.shape)"]),

code(["train_df.head(3)"]),

md(["## 3. Partición de los datos\n",
    "\n",
    "Mantenemos el mismo split de las iteraciones anteriores para que los scores sean comparables."]),

code(["X = train_df['text']\n",
      "y = train_df['decade']\n",
      "\n",
      "X_train, X_val, y_train, y_val = train_test_split(\n",
      "    X, y, test_size=0.2, random_state=1, stratify=y\n",
      ")\n",
      "\n",
      "print('Train:', X_train.shape[0], '| Val:', X_val.shape[0])"]),

md(["## 4. Diseño de experimentos\n",
    "\n",
    "Partimos del mejor modelo de la iteración 7 y exploramos sistemáticamente el efecto de `min_df`, el tamaño del vocabulario y el rango de n-grams."]),

code(["def build_char_model(c_value=0.40, ngram_range=(1, 7), max_features=250000,\n",
      "                     min_df=2, max_df=1.0):\n",
      "    return Pipeline([\n",
      "        ('tfidf', TfidfVectorizer(\n",
      "            lowercase=True,\n",
      "            strip_accents=None,\n",
      "            analyzer='char',\n",
      "            ngram_range=ngram_range,\n",
      "            min_df=min_df,\n",
      "            max_df=max_df,\n",
      "            sublinear_tf=True,\n",
      "            max_features=max_features,\n",
      "            dtype=np.float32,\n",
      "        )),\n",
      "        ('clf', LinearSVC(C=c_value)),\n",
      "    ])"]),

code(["candidate_models = {\n",
      "    'ref_iter7':     build_char_model(c_value=0.40, ngram_range=(1, 7), max_features=250000, min_df=2),\n",
      "    'mindf1_150k':   build_char_model(c_value=0.40, ngram_range=(1, 7), max_features=150000, min_df=1),\n",
      "    'mindf1_200k':   build_char_model(c_value=0.40, ngram_range=(1, 7), max_features=200000, min_df=1),\n",
      "    'mindf2_300k':   build_char_model(c_value=0.40, ngram_range=(1, 7), max_features=300000, min_df=2),\n",
      "    'mindf2_350k':   build_char_model(c_value=0.35, ngram_range=(1, 7), max_features=350000, min_df=2),\n",
      "}"]),

md(["## 5. Entrenamiento y comparación de candidatos"]),

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
      "    print(f'{name:<24} acc={acc:.4f}  f1={f1:.4f}  ({elapsed:.1f}s)')"]),

code(["results_df = pd.DataFrame(results).sort_values('val_accuracy', ascending=False).reset_index(drop=True)\n",
      "print(results_df.to_string(index=False))"]),

md(["Con el mejor rango y configuración identificados, afinamos el valor de `C`."]),

code(["best_name     = results_df.iloc[0]['modelo']\n",
      "best_pipeline = candidate_models[best_name]\n",
      "best_ngram    = best_pipeline.named_steps['tfidf'].ngram_range\n",
      "best_maxf     = best_pipeline.named_steps['tfidf'].max_features\n",
      "best_mindf    = best_pipeline.named_steps['tfidf'].min_df\n",
      "\n",
      "print(f'Mejor candidato base: {best_name}')\n",
      "print(f'  ngram_range={best_ngram}, max_features={best_maxf}, min_df={best_mindf}')"]),

code(["c_candidates = {}\n",
      "for c_val in [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:\n",
      "    name  = f'c_{int(c_val*100):03d}'\n",
      "    model = build_char_model(c_value=c_val, ngram_range=best_ngram,\n",
      "                              max_features=best_maxf, min_df=best_mindf)\n",
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

md(["## 6. Evaluación del mejor modelo"]),

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
      "plt.title('Matriz de confusión — validación (iter8)')\n",
      "plt.tight_layout()\n",
      "plt.show()"]),

md(["## 7. Predicciones sobre eval.csv\n",
    "\n",
    "Reentrenamos el mejor pipeline sobre el conjunto completo de entrenamiento antes de generar las predicciones finales."]),

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

path = 'c:/Users/jeron/OneDrive/Escritorio/Programacion/Aprendizaje-en-maquina/Proyectos/parte1_iteracion8.ipynb'
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f'Escrito: {path}')
