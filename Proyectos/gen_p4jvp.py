import json, random

def uid():
    return ''.join(random.choices('0123456789abcdef', k=8))

def md(src):
    return {"cell_type": "markdown", "id": uid(), "metadata": {}, "source": src}

def code(src):
    return {"cell_type": "code", "execution_count": None, "id": uid(), "metadata": {}, "outputs": [], "source": src}

cells = [

md(["# Character N-Grams con LinearSVC\n",
    "\n",
    "En este notebook implementamos el mejor enfoque encontrado para clasificar textos históricos en español por su década de origen. Se utilizan n-grams de caracteres como representación del texto, preservando toda la información ortográfica y morfológica del español histórico (siglos XVI–XIX). Como clasificador se usa una Máquina de Vectores de Soporte lineal (`LinearSVC`), que resulta especialmente eficiente para datos de alta dimensionalidad como los vectores TF-IDF de caracteres.\n",
    "\n",
    "El pipeline encapsula el vectorizador y el clasificador dentro de un objeto único, garantizando que la transformación TF-IDF se ajuste únicamente con los datos de entrenamiento de cada fold durante la validación cruzada y evitando así cualquier fuga de información."]),

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

md(["## 4. Preprocesamiento del texto\n",
    "\n",
    "A diferencia de los enfoques basados en palabras, aquí no aplicamos limpieza manual del texto. La vectorización se basa en caracteres (`analyzer='char'`), por lo que preservar la ortografía original —incluyendo tildes, la ñ y otros diacríticos— es fundamental: estos caracteres evolucionaron de forma distinta en cada siglo y constituyen señales temporales directas.\n",
    "\n",
    "La única normalización que realizamos es `lowercase=True` dentro del vectorizador. Se desactiva explícitamente el strip de acentos (`strip_accents=None`) para conservar toda esta información."]),

code(["df = df.drop_duplicates().reset_index(drop=True)\n",
      "\n",
      "print('Registros tras eliminar duplicados:', len(df))\n",
      "df[['text', 'decade']].head()"]),

md(["## 5. Partición de los datos"]),

code(["X = df['text']\n",
      "y = df['decade']\n",
      "\n",
      "X_train, X_val, y_train, y_val = train_test_split(\n",
      "    X, y, test_size=0.2, random_state=1, stratify=y\n",
      ")"]),

md(["Se usa `stratify=y` para mantener la proporción de clases en ambas particiones."]),

code(["X_train.shape, X_val.shape"]),

md(["## 6. Construcción del pipeline\n",
    "\n",
    "El pipeline encadena el vectorizador TF-IDF de caracteres con el clasificador `LinearSVC`. Al estar incluido dentro del pipeline, el vectorizador se ajusta únicamente sobre los datos de entrenamiento en cada partición interna de la validación cruzada, lo que garantiza que no hay fuga de información del conjunto de validación hacia el vocabulario ni hacia los pesos IDF.\n",
    "\n",
    "La configuración del vectorizador utiliza los hiperparámetros óptimos identificados en las iteraciones de experimentación: `ngram_range=(1,7)` captura desde caracteres individuales hasta secuencias de 7 caracteres, y `sublinear_tf=True` aplica escala logarítmica a las frecuencias de término para reducir el impacto de los n-grams muy frecuentes."]),

code(["pipeline_charsvm = Pipeline([\n",
      "    ('tfidf', TfidfVectorizer(\n",
      "        lowercase=True,\n",
      "        strip_accents=None,\n",
      "        analyzer='char',\n",
      "        ngram_range=(1, 7),\n",
      "        sublinear_tf=True,\n",
      "        max_features=300000,\n",
      "        dtype=np.float32,\n",
      "    )),\n",
      "    ('clf', LinearSVC()),\n",
      "])"]),

md(["## 7. Entrenamiento con búsqueda de hiperparámetros\n",
    "\n",
    "La búsqueda explora el parámetro de regularización `C` del clasificador y la frecuencia mínima de documento `min_df` del vectorizador. Un `C` pequeño impone mayor regularización y reduce el sobreajuste; `min_df=1` incluye n-grams que aparecen una sola vez en el corpus, lo que puede capturar patrones ortográficos muy específicos de una época."]),

code(["param_grid = {\n",
      "    'tfidf__min_df': [1, 2],\n",
      "    'clf__C': [0.20, 0.30, 0.40, 0.50],\n",
      "}"]),

code(["kfold = KFold(n_splits=3, shuffle=True, random_state=0)\n",
      "grid_charsvm = GridSearchCV(\n",
      "    pipeline_charsvm, param_grid, cv=kfold, scoring='accuracy', n_jobs=-1, verbose=1\n",
      ")"]),

code(["grid_charsvm.fit(X_train, y_train)"]),

code(["print('Mejores hiperparámetros:', grid_charsvm.best_params_)\n",
      "print('Mejor score CV (accuracy):', round(grid_charsvm.best_score_, 4))"]),

code(["resultados_cv = pd.DataFrame(grid_charsvm.cv_results_)\n",
      "cols = ['param_clf__C', 'param_tfidf__min_df', 'mean_test_score', 'std_test_score', 'rank_test_score']\n",
      "print(resultados_cv[cols].sort_values('rank_test_score').to_string(index=False))"]),

md(["## 8. Evaluación del mejor modelo\n",
    "\n",
    "Aplicamos el mejor pipeline encontrado por GridSearchCV sobre el conjunto de validación."]),

code(["best_model = grid_charsvm.best_estimator_\n",
      "\n",
      "y_pred_train = best_model.predict(X_train)\n",
      "y_pred_val   = best_model.predict(X_val)"]),

md(["#### Comparación de rendimientos sobre entrenamiento y validación"]),

code(["print('Accuracy en entrenamiento:', round(accuracy_score(y_train, y_pred_train), 4))\n",
      "print('Accuracy en validación:   ', round(accuracy_score(y_val,   y_pred_val),   4))\n",
      "print('Mejor score CV (accuracy):', round(grid_charsvm.best_score_,               4))"]),

code(["print(classification_report(y_val, y_pred_val))"]),

code(["fig, ax = plt.subplots(figsize=(16, 12))\n",
      "ConfusionMatrixDisplay.from_predictions(y_val, y_pred_val, ax=ax, colorbar=False)\n",
      "plt.title('Matriz de confusión — validación')\n",
      "plt.tight_layout()\n",
      "plt.show()"]),

md(["## 9. Predicciones sobre eval.csv\n",
    "\n",
    "Reentrenamos el mejor pipeline sobre el conjunto de entrenamiento completo antes de generar las predicciones finales. Esto aprovecha todos los datos disponibles y mejora la estimación del modelo."]),

code(["from sklearn.base import clone\n",
      "\n",
      "final_model = clone(best_model)\n",
      "final_model.fit(X, y)\n",
      "\n",
      "print('Modelo reentrenado sobre el conjunto completo.')"]),

code(["y_eval_pred = final_model.predict(df_eval['text']).astype(int)\n",
      "\n",
      "submission = pd.DataFrame({'id': df_eval['id'], 'answer': y_eval_pred})\n",
      "submission.to_csv('submission_char_svm.csv', index=False)\n",
      "\n",
      "print('Filas generadas:', len(submission))\n",
      "submission.head()"]),

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

path = 'c:/Users/jeron/OneDrive/Escritorio/Programacion/Aprendizaje-en-maquina/Proyectos/P_4JVP.ipynb'
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f'Escrito: {path}')
