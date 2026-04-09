# Proyecto: Clasificación de Textos Históricos en Español

Competencia Kaggle del curso Aprendizaje de Máquina. El objetivo es predecir la **década de origen** (clases 150–188, correspondientes a los siglos XVI–XIX) de fragmentos de textos históricos en español.

---

## Datos

| Archivo | Descripción |
|---------|-------------|
| `train.csv` | 31.403 registros con columnas `text` y `decade` |
| `eval.csv` | 3.490 registros con columnas `id` y `text` (sin etiqueta) |

- 39 clases (una por década)
- Distribución balanceada (~800 ejemplos por clase)
- Sin valores nulos; 34 duplicados eliminados antes del entrenamiento

---

## Configuración común a todos los modelos

Todos los notebooks siguen la misma estructura y comparten estas decisiones de diseño:

- **Partición:** 80% entrenamiento / 20% validación con `stratify=y` y `random_state=1`
- **Validación cruzada:** `KFold(n_splits=5, shuffle=True, random_state=0)`
- **Modelo base:** `LogisticRegression(max_iter=1000, solver='saga')`
- **Búsqueda de hiperparámetros:** `GridSearchCV` con `scoring='accuracy'`
- **Pipeline:** `sklearn.pipeline.Pipeline` que encapsula vectorización + clasificación

---

## Notebooks

### P_1JVP — TF-IDF Clásico
`submission_tfidf_clasico.csv`

**Enfoque:** Representación TF-IDF de palabras (unigramas). Sirve como línea base del proyecto.

**Preprocesamiento:**
- Minúsculas
- Eliminación de caracteres que no sean letras del español (`[^a-záéíóúüñ\s]`)
- Normalización de espacios
- Eliminación de duplicados

Este nivel de limpieza reduce ruido tipográfico de la digitalización (números de página, símbolos) sin destruir el vocabulario arcaico que es informativo para clasificar épocas.

**Pipeline:** `TfidfVectorizer()` → `LogisticRegression`

**Hiperparámetros explorados:**
- `tfidf__max_features`: [20000, 50000]
- `tfidf__min_df`: [1, 2]
- `tfidf__max_df`: [0.95]
- `clf__C`: [0.1, 1, 10]

---

### P_1JZE — TF-IDF con N-Grams de Palabras
`submission_tfidf_ngrams.csv`

**Enfoque:** TF-IDF de palabras con n-grams extendidos (unigramas, bigramas y trigramas). Permite capturar expresiones y estructuras sintácticas características de distintas épocas del español.

**Preprocesamiento:** Idéntico al notebook anterior (limpieza completa de caracteres no alfabéticos).

**Pipeline:** `TfidfVectorizer()` → `LogisticRegression`

**Hiperparámetros explorados:**
- `tfidf__ngram_range`: [(1,1), (1,2), (1,3)]
- `tfidf__max_features`: [30000, 50000]
- `tfidf__min_df`: [1, 2]
- `clf__C`: [0.1, 1, 10]

---

### P_2JVP — Character N-Grams
`submission_char_ngrams.csv`

**Enfoque:** TF-IDF basado en secuencias de caracteres. Los n-grams de caracteres capturan patrones ortográficos que evolucionaron entre los siglos XVI y XIX: grafías arcaicas (p. ej. *f* por *h* inicial, *u/v* intercambiables), terminaciones verbales y formas morfológicas propias de cada época. El modelo no necesita entender palabras completas para detectar estos patrones.

**Preprocesamiento:** Solo minúsculas y normalización de espacios. Se conserva la puntuación y caracteres especiales porque pueden ser informativos para el análisis ortográfico.

**Pipeline:** `TfidfVectorizer(analyzer='char_wb')` → `LogisticRegression`

Se usa `analyzer='char_wb'` en lugar de `'char'` para que los n-grams no crucen los límites de palabra de forma arbitraria.

**Hiperparámetros explorados:**
- `tfidf__ngram_range`: [(2,4), (3,5), (3,6)]
- `tfidf__max_features`: [30000, 50000]
- `tfidf__min_df`: [2, 3]
- `clf__C`: [0.1, 1, 10]

---

### P_2JZE — Word + Character Combinado
`submission_word_char.csv`

**Enfoque:** Combina en un solo vector las representaciones TF-IDF de palabras y de caracteres usando `FeatureUnion`. El clasificador recibe información de ambas fuentes simultáneamente: el vocabulario de palabras aporta contenido semántico y los n-grams de caracteres aportan información ortográfica y morfológica.

**Preprocesamiento:** Solo minúsculas y normalización de espacios, para no perjudicar el componente de caracteres.

**Pipeline:** `FeatureUnion([TfidfVectorizer(word), TfidfVectorizer(char_wb)])` → `LogisticRegression`

**Hiperparámetros explorados:**
- `features__word__ngram_range`: [(1,1), (1,2)]
- `features__word__max_features`: [20000, 30000]
- `features__char__ngram_range`: [(3,5)]
- `features__char__max_features`: [20000, 30000]
- `clf__C`: [0.1, 1, 10]

---

### P_3JVP — Lemmatización
`submission_lemmatization.csv`

**Enfoque:** Lematización con spaCy (`es_core_news_sm`) antes de la vectorización TF-IDF. La lematización reduce cada palabra a su forma canónica según el diccionario morfológico (p. ej. *corriendo* → *correr*, *ciudades* → *ciudad*). Esto agrupa variantes morfológicas bajo un mismo token, reduciendo la dimensionalidad del vocabulario.

> **Nota:** requiere `python -m spacy download es_core_news_sm`

**Preprocesamiento:**
- Minúsculas
- Lematización con spaCy

El efecto en textos históricos puede ser parcial, ya que el modelo de spaCy fue entrenado con español moderno y puede no reconocer correctamente formas arcaicas.

**Pipeline:** El lema se calcula antes de la partición y se almacena en la columna `texto_lem`. El pipeline aplica `TfidfVectorizer()` → `LogisticRegression`.

**Hiperparámetros explorados:**
- `tfidf__ngram_range`: [(1,1), (1,2)]
- `tfidf__max_features`: [20000, 50000]
- `tfidf__min_df`: [1, 2]
- `clf__C`: [0.1, 1, 10]

---

### P_3JZE — Stemming
`submission_stemming.csv`

**Enfoque:** Stemming con `SnowballStemmer('spanish')` de NLTK antes de la vectorización TF-IDF. El stemming recorta los sufijos de cada palabra según reglas morfológicas heurísticas (p. ej. *corriendo*, *corrió*, *correr* → *corr*). Es más agresivo que la lematización pero más rápido y sin necesidad de modelo externo.

**Preprocesamiento:**
- Minúsculas
- Eliminación de caracteres no alfabéticos del español
- Stemming token a token

**Pipeline:** El texto stemmeado se calcula antes de la partición y se almacena en `texto_stem`. El pipeline aplica `TfidfVectorizer()` → `LogisticRegression`.

**Hiperparámetros explorados:** Idénticos al notebook de lematización para facilitar la comparación directa entre ambos enfoques de normalización:
- `tfidf__ngram_range`: [(1,1), (1,2)]
- `tfidf__max_features`: [20000, 50000]
- `tfidf__min_df`: [1, 2]
- `clf__C`: [0.1, 1, 10]

---

## Archivos de submission generados

| Archivo | Notebook |
|---------|---------|
| `submission_tfidf_clasico.csv` | P_1JVP |
| `submission_tfidf_ngrams.csv` | P_1JZE |
| `submission_char_ngrams.csv` | P_2JVP |
| `submission_word_char.csv` | P_2JZE |
| `submission_lemmatization.csv` | P_3JVP |
| `submission_stemming.csv` | P_3JZE |

Cada archivo tiene el formato requerido por Kaggle:
```
id,answer
2,164
5,172
...
```
