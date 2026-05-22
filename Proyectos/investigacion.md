# Estrategias Avanzadas de Modelado para la Clasificación Decenal de Textos Históricos en Español

La clasificación de documentos históricos mediante técnicas de aprendizaje profundo representa un desafío multidimensional que trasciende la simple categorización temática. En el caso específico de la determinación de la década de origen para textos que abarcan desde el año 1500 hasta el 1889, el investigador se enfrenta a una evolución lingüística continua, a la degradación de los soportes físicos traducida en errores de Reconocimiento Óptico de Caracteres (OCR) y a la transición de normativas ortográficas que no se estabilizaron hasta bien entrado el siglo XIX. Con un estado del arte actual situado en un score de 0.31518, la optimización del rendimiento exige un enfoque que integre la robustez de los modelos a nivel de bytes, la precisión de los Transformers preentrenados en español masivo y la lógica matemática de la regresión ordinal.

## Resumen Ejecutivo de Recomendaciones Técnicas

- **Adopción de Arquitecturas Token-free**: Se recomienda prioritariamente el uso de modelos como ByT5 o CANINE, los cuales operan directamente sobre bytes o caracteres, eliminando la degradación de señal producida por tokenizadores estándar que fragmentan erróneamente las grafías históricas y el ruido de OCR.<sup>1</sup>

- **Preentrenamiento Adaptativo al Dominio (DAPT)**: Es imperativo realizar una fase de preentrenamiento continuado utilizando Masked Language Modeling (MLM) exclusivamente sobre el conjunto de train.csv para alinear las representaciones vectoriales con el léxico de los siglos XVI al XIX.<sup>3</sup>

- **Optimización de Pérdida Ordinal**: Dado que las décadas poseen una estructura jerárquica y secuencial, la implementación de funciones de pérdida como CORAL (Consistent Rank Logits) o la Distancia de Earth Mover (EMD) resulta superior a la entropía cruzada convencional al penalizar proporcionalmente la distancia temporal del error.<sup>6</sup>

- **Uso de Modelos Preentrenados en Español**: Los modelos de la familia MarIA (RoBERTa-base-BNE) y BERTIN ofrecen la base de conocimiento lingüístico más profunda para el español, superando significativamente a las variantes multilingües en tareas de comprensión lectora y clasificación.<sup>9</sup>

- **Aumentación de Datos con Simulación de OCR**: La generación de muestras sintéticas que repliquen confusiones históricas (como la "s larga" por "f") mediante la biblioteca nlpaug incrementará la resiliencia del modelo ante textos ruidosos.<sup>12</sup>

- **Gestión de Contexto Largo**: Para los fragmentos que superan el límite de 512 tokens, se debe implementar una estrategia de *sliding window* con *attention pooling* o utilizar modelos de contexto extendido como Longformer.<sup>9</sup>

- **Preservación de Grafías Históricas**: La normalización agresiva debe evitarse; marcadores como la alternancia "u/v" o "i/j" son señales temporales críticas que el modelo debe aprender a interpretar en lugar de eliminar.

- **Label Smoothing Unimodal**: El suavizado de etiquetas hacia las décadas adyacentes ayuda a regularizar el modelo y refleja la naturaleza transicional de los cambios lingüísticos.<sup>17</sup>

- **Validación Cruzada Estratificada**: Se requiere un control estricto para evitar que fragmentos de una misma obra o autor se filtren entre los conjuntos de entrenamiento y validación, previniendo el sobreajuste estilístico.

- **Ensembles Híbridos**: La combinación de la potencia semántica de los Transformers con la robustez de los n-gramas de caracteres (TF-IDF) mediante técnicas de *stacking* proporciona una mejora consistente en la métrica final.

- **Integración de Datos Externos**: El enriquecimiento con textos de Project Gutenberg o la Biblioteca Virtual Miguel de Cervantes puede equilibrar las clases y proporcionar contextos más limpios para el preentrenamiento.<sup>20</sup>

- **Ajuste Fino de Hiperparámetros**: El uso de tasas de aprendizaje diferenciadas y técnicas de *mixed precision* (FP16) es esencial para optimizar el entrenamiento en infraestructuras de GPU única.<sup>23</sup>

## Análisis Comparativo de Estrategias y Modelos

La siguiente tabla presenta una comparativa técnica detallada de las estrategias propuestas, evaluando su impacto potencial frente al costo de implementación y computación.

| **Estrategia / Modelo** | **Justificación Técnica** | **Riesgo Principal** | **Costo Computacional** | **Prioridad** | **Evidencia / Fuente** |
| --- | --- | --- | --- | --- | --- |
| **ByT5-Base** | Resiliencia intrínseca a errores de OCR y variantes ortográficas mediante procesamiento de bytes. | Secuencias más largas que aumentan el uso de memoria y tiempo de cómputo. | Alto | **Crítica** | 1 |
| **MarIA (RoBERTa-BNE)** | Preentrenado con 570GB de español limpio; máxima capacidad semántica en el idioma. | Sesgo hacia el español moderno; requiere ajuste de dominio para el siglo XVI. | Medio | **Alta** | 9 |
| **CORAL / CORN Loss** | Garantiza la consistencia de rango en predicciones ordinales; reduce el error absoluto medio (MAE). | Requiere modificaciones en la arquitectura de la capa de salida. | Bajo | **Alta** | 6 |
| **DAPT (MLM en Train)** | Adapta el vocabulario y pesos del modelo al español histórico y al ruido específico del dataset. | Riesgo de olvido catastrófico si no se monitoriza la validación. | Medio-Alto | **Alta** | 3 |
| **Longformer (4096 tokens)** | Permite procesar textos completos sin truncamiento, capturando señales bibliográficas finales. | Poca ganancia si la señal temporal está concentrada al inicio. | Alto | **Media** | 9 |
| **Label Smoothing Ordinal** | Representa la incertidumbre temporal; un texto de 1605 puede parecer de 1595. | Puede degradar el Accuracy si el factor de suavizado es excesivo. | Muy Bajo | **Alta** | 17 |
| **nlpaug (OCR Noise)** | Fortalece la robustez mediante la simulación de degradación física y errores de escaneo. | El ruido sintético puede no coincidir perfectamente con el ruido real del dataset. | Bajo | **Media** | 12 |
| **Attention Pooling** | Agrega información de múltiples ventanas de texto priorizando fragmentos relevantes. | Mayor complejidad en la implementación del bucle de entrenamiento. | Bajo | **Media** | 15 |

## Evolución de los Modelos de Lenguaje en el Contexto Histórico

El procesamiento de textos históricos requiere una comprensión de cómo los modelos de lenguaje modernos interpretan la evolución de la gramática y la ortografía. El español de los siglos XVI y XVII, a menudo denominado español medio o áureo, presenta divergencias estructurales con el español contemporáneo que afectan la eficiencia de los tokenizadores de subpalabras (como BPE o WordPiece). Cuando un modelo como BETO o MarIA procesa términos como "dixo" (dijo) o "hazer" (hacer), el tokenizador puede descomponer estas palabras en fragmentos que carecen de la carga semántica original, perdiendo así la señal temporal que reside en la propia grafía.<sup>10</sup>

Los modelos *token-free* o basados en bytes, representados principalmente por ByT5, ofrecen una solución disruptiva a este problema. Al no depender de un vocabulario predefinido, ByT5 es capaz de aprender que la secuencia de bytes correspondiente a "vna" es funcionalmente equivalente a "una", pero que la primera tiene una probabilidad de ocurrencia drásticamente superior en la década de 1550 que en la de 1880.<sup>1</sup> Investigaciones recientes demuestran que ByT5 reduce significativamente el Character Error Rate (CER) en tareas de corrección de OCR histórico, logrando mejoras de hasta el 50% en la precisión de recuperación de texto original frente a modelos basados en tokens.<sup>25</sup>

Para el corpus de esta competencia, que presenta un 50% de vocabulario único en el conjunto de evaluación no visto en el entrenamiento, la capacidad de generalización de ByT5 es fundamental. Mientras que los modelos tradicionales asignarían vectores de "desconocido" (UNK) o fragmentarían excesivamente las palabras raras, ByT5 construye representaciones basadas en la morfología de los bytes, permitiéndole identificar sufijos y raíces históricas incluso en palabras que no aparecieron explícitamente en la fase de preentrenamiento.

## Modelos Preentrenados y su Adaptación al Dominio

La elección del modelo base es el factor determinante en el éxito de la transferencia de aprendizaje. En el panorama del español, existen tres pilares fundamentales:

- **MarIA (RoBERTa-base-BNE)**: Desarrollado por el Barcelona Supercomputing Center, este modelo fue entrenado con 570GB de texto proveniente de los archivos de la Biblioteca Nacional de España entre 2009 y 2019.<sup>9</sup> Aunque su base es moderna, su escala le otorga una comprensión sintáctica del español sin parangón. Sin embargo, para su aplicación en el siglo XVI, es imperativo realizar una fase de *Domain-Adaptive Pre-training* (DAPT).<sup>3</sup>

- **BERTIN**: Este modelo utiliza una técnica innovadora de *perplexity sampling* para seleccionar los mejores datos del corpus mC4 español.<sup>32</sup> Se ha demostrado que BERTIN es altamente eficiente en tareas de clasificación general, superando a menudo a mBERT y a veces compitiendo con MarIA en contextos de recursos limitados.<sup>32</sup>

- **BETO**: El pionero de los modelos de BERT en español, entrenado por la Universidad de Chile, sigue siendo una opción robusta para tareas de clasificación de nivel base debido a su entrenamiento con Whole Word Masking (WWM).<sup>30</sup>

La estrategia de DAPT consiste en continuar el entrenamiento de estos modelos sobre el corpus de la competencia utilizando el objetivo de MLM. Esto permite que el modelo "vea" el ruido de OCR y las grafías arcaicas antes de enfrentarse a la tarea de clasificación supervisada. Se ha observado que incluso una sola época de DAPT sobre un pequeño conjunto de datos específicos del dominio puede mejorar el F1-score en varios puntos, al reducir la pérdida por desconexión de vocabulario.<sup>3</sup> En este contexto, el uso de MLM en el conjunto de entrenamiento de 31,403 textos proporcionará al modelo una familiaridad crítica con términos como "vuestra merced", "fobre" o "cofa", que de otro modo podrían ser interpretados como errores aleatorios.<sup>27</sup>

## Estrategias para Textos con Ruido y Longitud Extendida

El dataset presenta una mediana de 50 palabras, pero con una cola larga que alcanza las 900 palabras y los 1,600 caracteres. Un truncamiento ingenuo a 512 tokens (límite estándar de BERT) podría descartar información vital. En textos legales o administrativos de los siglos XVIII y XIX, las fechas o referencias institucionales específicas suelen aparecer al final de los párrafos largos.

### Modelado de Contexto Largo

Para mitigar la pérdida de información por truncamiento, se proponen tres aproximaciones técnicas:

- **Tratamiento de Ventanas Deslizantes (Sliding Window)**: Consiste en dividir el fragmento en bloques solapados (e.g., de 512 tokens con un solape de 128). Cada bloque se procesa de forma independiente y sus representaciones finales se combinan mediante una capa de atención o un simple promedio de probabilidades.<sup>15</sup>

- **Longformer**: Esta arquitectura sustituye la atención cuadrática por una atención local con "ventanas" y atención global en tokens específicos (como el token de clasificación `[CLS]`). El modelo longformer-base-4096-bne-es es la variante recomendada, permitiendo procesar hasta 4096 tokens y manteniendo la coherencia semántica de textos extensos.<sup>9</sup>

- **Hierarchical Attention Networks (HAN)**: Una alternativa deep no-transformer que procesa el texto a nivel de palabras y luego de oraciones, aplicando atención en ambos niveles para identificar qué términos o frases son más indicativos de una década específica.

### Robustez ante el Ruido de OCR

El ruido de OCR es una variable constante en el dataset. La literatura sugiere que la mejor defensa es una buena ofensa: la aumentación de datos reactiva. Utilizando la biblioteca nlpaug, es posible simular la degradación física de los textos históricos.<sup>12</sup>

- **Confusión de Caracteres**: Implementar mapeos de error comunes en escaneos antiguos, como la confusión entre 'f' y 'ſ' (s larga), o entre 'u' y 'v'.<sup>13</sup>

- **Inserción de Ruido**: Añadir saltos de línea erróneos o caracteres especiales que suelen aparecer en transcripciones imperfectas.

- **Back-translation**: Traducir el español histórico a un idioma moderno intermedio (e.g., alemán o francés) y volver al español puede ayudar a parafrasear el texto, aunque se debe manejar con precaución para no modernizar excesivamente el estilo y perder la señal temporal.<sup>14</sup>

## Fundamentos Matemáticos de la Regresión Ordinal en Deep Learning

La tarea de predecir décadas es intrínsecamente ordinal. Un error de predicción de una década (e.g., predecir 1600 en lugar de 1610) debe penalizarse menos que un error de tres siglos. La entropía cruzada tradicional trata cada década como una categoría aislada, ignorando esta relación métrica.

### Pérdida CORAL y CORN

El enfoque CORAL (Consistent Rank Logits) propone descomponer el problema de $K$ clases en $K-1$ subtareas binarias. Para una década objetivo, el modelo aprende a predecir si el texto es posterior a la década 150, posterior a la 151, y así sucesivamente. Esto garantiza la consistencia del rango: si un texto tiene una probabilidad alta de pertenecer a una década posterior a 1700, necesariamente debe tener una probabilidad aún mayor de ser posterior a 1600.<sup>6</sup> La implementación de CORAL ha demostrado reducir drásticamente el error absoluto medio (MAE) en tareas de estimación de edad a partir de imágenes y clasificación de severidad en medicina.<sup>6</sup>

### Distancia de Earth Mover (EMD)

La pérdida basada en EMD minimiza el costo de transformar la distribución de probabilidad predicha en la distribución real. Para variables ordinales, esto se traduce en minimizar la diferencia entre las Funciones de Distribución Acumulada (CDF).<sup>7</sup> La formulación cuadrática de la EMD es especialmente eficaz en redes neuronales debido a su diferenciabilidad suave:

$$\mathcal{L}_{EMD} = \sum_{k=1}^{K} \left( CDF_{pred}(k) - CDF_{real}(k) \right)^2$$

Esta pérdida obliga al modelo a asignar probabilidades a clases cercanas a la real cuando tiene incertidumbre, en lugar de dispersar la probabilidad de forma aleatoria por todo el espectro temporal.<sup>8</sup>

### Label Smoothing Unimodal

En lugar de una etiqueta de "talla única" (one-hot encoding), se recomienda suavizar la etiqueta real siguiendo una distribución beta o gaussiana centrada en la década correcta. Por ejemplo, para un texto de la década 160 (1600-1609), la etiqueta objetivo podría asignar un peso de 0.7 a la clase 160, 0.12 a las clases 159 y 161, y 0.03 a las clases 158 y 162.<sup>17</sup> Esto refleja la realidad histórica de que los cambios lingüísticos no ocurren abruptamente en la frontera de una década.<sup>19</sup>

## Plan Experimental Priorizado

Se propone un ciclo de 6 experimentos diseñados para superar incrementalmente el score actual, comenzando por las intervenciones de mayor impacto y menor costo de implementación.

### Experimento 1: Adaptación de Dominio y ByT5-Base

- **Hipótesis**: Un modelo que procesa bytes directamente y es preentrenado con los textos de la competencia superará la fragmentación del tokenizador de subpalabras en español antiguo.

- **Modelo**: google/byt5-base.

- **Preprocesamiento**: Limpieza mínima; preservación de puntuación original y grafías.

- **Aumentación**: Ninguna para establecer un baseline sólido.

- **Validación**: Stratified K-Fold (K=5).

- **Criterio de Éxito**: Superar el Accuracy local de 0.315.

- **Tiempo Estimado**: 12 horas en GPU T4.

### Experimento 2: RoBERTa-BNE con Pérdida CORAL

- **Hipótesis**: La restricción ordinal de CORAL reducirá el error de "predicciones lejanas", mejorando la coherencia temporal del modelo.

- **Modelo**: PlanTL-GOB-ES/roberta-base-bne.

- **Preprocesamiento**: Normalización de espacios y saltos de línea.

- **Aumentación**: Character Dropout (0.02) durante el entrenamiento.

- **Criterio de Éxito**: Reducción del MAE temporal en un 15% respecto al baseline de TF-IDF.

- **Tiempo Estimado**: 8 horas en GPU T4.

### Experimento 3: Inyección de Ruido OCR y Aumentación Sintética

- **Hipótesis**: Entrenar con ruido controlado permitirá al modelo ignorar artefactos de escaneo y centrarse en la estructura lingüística subyacente.

- **Modelo**: Mejor modelo del Exp. 1 o 2.

- **Aumentación**: Pipeline de nlpaug con KeyboardAug y OcrAug personalizado para español antiguo.

- **Criterio de Éxito**: Mejora de +0.015 en el score público de Kaggle.

- **Tiempo Estimado**: 10 horas en GPU T4.

### Experimento 4: Entrenamiento con Suavizado de Etiquetas Gaussiano

- **Hipótesis**: El suavizado de etiquetas mejorará la calibración del modelo en las décadas limítrofes, reduciendo el sobreajuste a fechas específicas.

- **Técnica**: Label Smoothing Unimodal con $\sigma = 1.5$ décadas.

- **Criterio de Éxito**: Mejora en la métrica de log-loss y mayor robustez en validación cruzada.

- **Tiempo Estimado**: 6 horas en GPU T4.

### Experimento 5: Estrategia de Contexto Largo con Ventanas Deslizantes

- **Hipótesis**: Capturar información de los párrafos largos mediante ventanas deslizantes y attention pooling mejorará la predicción de documentos administrativos extensos.

- **Modelo**: MarIA-RoBERTa con capa de agregación jerárquica.

- **Criterio de Éxito**: Mejora significativa en el Accuracy del percentil 95 de longitud (textos > 1600 caracteres).

- **Tiempo Estimado**: 15 horas en GPU T4.

### Experimento 6: Ensemble de Stacking (Deep + Char N-grams)

- **Hipótesis**: Los modelos de aprendizaje profundo y los modelos estadísticos de n-gramas de caracteres cometen errores en diferentes tipos de textos; su combinación compensará estas debilidades.

- **Técnica**: Logistic Regression como meta-clasificador sobre las probabilidades de ByT5, MarIA y TF-IDF.

- **Criterio de Éxito**: Superar el umbral de 0.35 en Kaggle.

- **Tiempo Estimado**: 4 horas (post-procesamiento de predicciones previas).

## Configuración Inicial Recomendada para GPU Única

Para el primer intento de modelado profundo bajo restricciones de hardware (1 GPU de 12GB-16GB VRAM), se recomienda la siguiente arquitectura y configuración de hiperparámetros:

- **Modelo Base**: PlanTL-GOB-ES/roberta-base-bne (por su balance entre rendimiento y consumo de memoria).<sup>9</sup>

- **Cabeza de Clasificación**: Capa lineal de 768 unidades a $K-1$ (38) salidas binarias para implementar el marco CORAL.<sup>6</sup>

- **Tokenización**: max_length=512, con truncamiento inteligente (primeros 256 + últimos 256 tokens) para capturar el inicio y el cierre del fragmento.

- **Optimizador**: AdamW con learning_rate=2e-5, weight_decay=0.01 y un esquema de linear_schedule_with_warmup durante el primer 10% de los pasos.<sup>23</sup>

- **Regularización**: Dropout de 0.1 en las capas ocultas y label_smoothing=0.1.<sup>23</sup>

- **Entrenamiento**: 5 a 8 épocas, con batch_size=16 y acumulación de gradientes de 2 pasos si la memoria es insuficiente.

- **Hardware**: Habilitar mixed_precision='fp16' para reducir el uso de VRAM y acelerar el cómputo en GPUs modernas.<sup>23</sup>

## Análisis de Riesgos y Mitigación

El desarrollo de un clasificador de alta precisión para esta tarea conlleva riesgos técnicos inherentes que deben ser gestionados de forma proactiva:

- **Fuga de Datos (Leakage)**: Dado que el dataset contiene duplicados y fragmentos de las mismas obras, existe el riesgo de que el modelo aprenda a identificar el estilo específico de un autor en lugar de la señal temporal. La validación debe realizarse agrupando textos por similitud léxica extrema (Jaccard similarity > 0.9) para asegurar que muestras casi idénticas no se compartan entre train y validación.

- **Sobreajuste a Grafías Modernizadas**: Algunos textos pueden haber sido editados en siglos posteriores, modernizando su ortografía. Si el modelo se apoya excesivamente en la ortografía ("vna" vs "una"), fallará en textos antiguos modernizados. La mitigación consiste en entrenar el modelo tanto con el texto original como con versiones aumentadas donde se modernicen o arcaicen términos de forma aleatoria.

- **Mala Calibración de Probabilidades**: Los Transformers suelen ser demasiado confiados en sus predicciones. El uso de Temperature Scaling en el conjunto de validación antes de realizar el ensemble es crucial para asegurar que las probabilidades reflejen la incertidumbre real de la datación.

- **Deriva del Leaderboard (Leaderboard Drift)**: Con solo 3,490 textos en evaluación, el azar puede jugar un papel importante en las mejoras de la tabla pública. Se debe dar prioridad absoluta a la métrica de validación cruzada ($\bar{CV}$) y al MAE local. Un modelo con un Accuracy marginalmente menor pero un MAE significativamente más bajo es generalmente más robusto para el Private Leaderboard.<sup>7</sup>

- **Dependencia de Datos Externos**: Si se utilizan datos de Project Gutenberg o la BDH, existe el riesgo de introducir un *domain shift* si esos textos tienen un OCR mucho más limpio que el dataset de la competencia. Se recomienda añadir ruido sintético a los datos externos para igualar su calidad a la de train.csv.<sup>2</sup>

## Fuentes de Datos Externos y Licencias

La integración de datos externos es una estrategia permitida y recomendada siempre que se respeten las licencias originales:

- **Biblioteca Digital Hispánica (BNE)**: Proporciona acceso a millones de páginas digitalizadas. El corpus utilizado para preentrenar MarIA (570GB) ya incluye gran parte de esta información, pero el acceso a colecciones específicas de los siglos XVI al XVIII puede ser útil para tareas de MLM.<sup>22</sup>

- **Project Gutenberg**: Alberga miles de obras en español del siglo XIX y clásicos del Siglo de Oro. Licencia libre para fines de investigación.<sup>20</sup>

- **Corpus CODEA**: Especializado en documentos administrativos y legales antiguos, ideal para equilibrar el sesgo literario que suelen tener los datasets históricos.<sup>45</sup>

- **Wikisource**: Repositorio de textos en dominio público que incluye transcripciones manuales, a menudo más limpias que el OCR crudo, útiles para crear muestras de "español histórico limpio" para técnicas de destilación o aumentación.<sup>21</sup>

## Matriz de Decisiones Inmediatas

Antes de iniciar la siguiente iteración de modelado, se deben tomar las siguientes decisiones estratégicas:

- **Modelo de Base**: ¿Comenzar con ByT5 para máxima robustez ante ruido o con MarIA para máxima potencia semántica? Recomendación: ByT5 para establecer el techo de rendimiento inicial.

- **Métrica Primaria de Optimización**: ¿Accuracy o MAE? Recomendación: Optimizar para MAE mediante pérdidas ordinales (CORAL/EMD) y monitorear Accuracy.

- **Nivel de Preprocesamiento**: ¿Normalizar "ſ" a "s" y "v" a "u" de forma sistemática? Decisión: No. Mantener el ruido original y usar preentrenamiento MLM para que el modelo aprenda las equivalencias.

- **Estrategia de Validación**: Definir una semilla única para los splits de K-Fold y no modificarla para asegurar la comparabilidad entre experimentos.

- **Hardware**: ¿Uso de Gradiente Acumulado para simular batches grandes? Decisión: Sí, es esencial para la estabilidad del entrenamiento de Transformers en GPUs de memoria media.

- **Uso de Datos Externos**: ¿Integrar datos de Gutenberg desde el día 1? Decisión: No. Primero agotar el potencial de train.csv mediante DAPT/MLM y luego añadir datos externos si las curvas de aprendizaje sugieren falta de diversidad.

Esta investigación proporciona una hoja de ruta técnica exhaustiva, fundamentada en la intersección del NLP moderno y la lingüística histórica, para elevar el rendimiento del modelo actual hacia niveles competitivos en la clasificación decenal de textos.

---

#### Fuentes citadas

- ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models - ACL Anthology, acceso: mayo 13, 2026, [https://aclanthology.org/2022.tacl-1.17/](https://aclanthology.org/2022.tacl-1.17/)

- OCR Correction with ByT5. We have trained a Dutch OCR correction ..., acceso: mayo 13, 2026, [https://blog.ml6.eu/ocr-correction-with-byt5-5994d1217c07](https://blog.ml6.eu/ocr-correction-with-byt5-5994d1217c07)

- Domain-Adaptive Pre-Training (DAPT) - Emergent Mind, acceso: mayo 13, 2026, [https://www.emergentmind.com/topics/domain-adaptive-pre-training-dapt](https://www.emergentmind.com/topics/domain-adaptive-pre-training-dapt)

- Efficient Domain-adaptive Continual Pretraining for the Process Industry in the German Language - ResearchGate, acceso: mayo 13, 2026, [https://www.researchgate.net/publication/391246823_Efficient_Domain-adaptive_Continual_Pretraining_for_the_Process_Industry_in_the_German_Language](https://www.researchgate.net/publication/391246823_Efficient_Domain-adaptive_Continual_Pretraining_for_the_Process_Industry_in_the_German_Language)

- Efficient Domain-adaptive Continual Pretraining for the Process Industry in the German Language - arXiv, acceso: mayo 13, 2026, [https://arxiv.org/html/2504.19856v1](https://arxiv.org/html/2504.19856v1)

- GitHub - Raschka-research-group/coral-pytorch: CORAL and CORN ..., acceso: mayo 13, 2026, [https://github.com/Raschka-research-group/coral-pytorch](https://github.com/Raschka-research-group/coral-pytorch)

- A Simple Log-based Loss Function for Ordinal Text ... - ACL Anthology, acceso: mayo 13, 2026, [https://aclanthology.org/2022.coling-1.407.pdf](https://aclanthology.org/2022.coling-1.407.pdf)

- Squared Earth Mover's Distance Loss for Training Deep Neural Networks on Ordered-Classes - Computer Science, acceso: mayo 13, 2026, [https://www3.cs.stonybrook.edu/~cvl/content/papers/2017/Hou_NIPSW17.pdf](https://www3.cs.stonybrook.edu/~cvl/content/papers/2017/Hou_NIPSW17.pdf)

- PlanTL-GOB-ES/lm-spanish: Official source for spanish Language Models and resources made @ BSC-TEMU within the "Plan de las Tecnologías del Lenguaje" (Plan-TL). - GitHub, acceso: mayo 13, 2026, [https://github.com/PlanTL-GOB-ES/lm-spanish](https://github.com/PlanTL-GOB-ES/lm-spanish)

- MarIA: Spanish Language Models - arXiv, acceso: mayo 13, 2026, [https://arxiv.org/pdf/2107.07253](https://arxiv.org/pdf/2107.07253)

- PlanTL-GOB-ES/roberta-base-bne - Hugging Face, acceso: mayo 13, 2026, [https://huggingface.co/PlanTL-GOB-ES/roberta-base-bne](https://huggingface.co/PlanTL-GOB-ES/roberta-base-bne)

- Text Generation|nlpaug| - Kaggle, acceso: mayo 13, 2026, [https://www.kaggle.com/code/ilyazored/text-generation-nlpaug](https://www.kaggle.com/code/ilyazored/text-generation-nlpaug)

- nlpaug — nlpaug 1.1.11 documentation, acceso: mayo 13, 2026, [https://nlpaug.readthedocs.io/](https://nlpaug.readthedocs.io/)

- NLPAUG - A Python library to Augment Your Text Data - Analytics Vidhya, acceso: mayo 13, 2026, [https://www.analyticsvidhya.com/blog/2021/08/nlpaug-a-python-library-to-augment-your-text-data/](https://www.analyticsvidhya.com/blog/2021/08/nlpaug-a-python-library-to-augment-your-text-data/)

- Multi-Level Attention Pooling (MLAP) - Emergent Mind, acceso: mayo 13, 2026, [https://www.emergentmind.com/topics/multi-level-attention-pooling-mlap](https://www.emergentmind.com/topics/multi-level-attention-pooling-mlap)

- PlanTL-GOB-ES/longformer-base-4096-bne-es - Hugging Face, acceso: mayo 13, 2026, [https://huggingface.co/PlanTL-GOB-ES/longformer-base-4096-bne-es](https://huggingface.co/PlanTL-GOB-ES/longformer-base-4096-bne-es)

- The unimodal model for the classification of ordinal data | Request PDF - ResearchGate, acceso: mayo 13, 2026, [https://www.researchgate.net/publication/5754853_The_unimodal_model_for_the_classification_of_ordinal_data](https://www.researchgate.net/publication/5754853_The_unimodal_model_for_the_classification_of_ordinal_data)

- Predicting Customer Satisfaction with Soft Labels for Ordinal Classification - ACL Anthology, acceso: mayo 13, 2026, [https://aclanthology.org/2023.acl-industry.62.pdf](https://aclanthology.org/2023.acl-industry.62.pdf)

- Improving Regression Performance with Distributional Losses - Proceedings of Machine Learning Research, acceso: mayo 13, 2026, [http://proceedings.mlr.press/v80/imani18a/imani18a.pdf](http://proceedings.mlr.press/v80/imani18a/imani18a.pdf)

- Books by Cervantes Saavedra, Miguel de - Project Gutenberg, acceso: mayo 13, 2026, [http://www.gutenberg.org/ebooks/author/505](http://www.gutenberg.org/ebooks/author/505)

- Miguel de Cervantes - Wikisource, the free online library, acceso: mayo 13, 2026, [https://en.wikisource.org/wiki/Author:Miguel_de_Cervantes](https://en.wikisource.org/wiki/Author:Miguel_de_Cervantes)

- Next generation of metadata in cultural heritage: continuing the conversation in Spain, acceso: mayo 13, 2026, [https://hangingtogether.org/next-generation-of-metadata-in-cultural-heritage-continuing-the-conversation-in-spain/](https://hangingtogether.org/next-generation-of-metadata-in-cultural-heritage-continuing-the-conversation-in-spain/)

- Fine-Tuning BERT for Classification: A Practical Guide | by Hey Amit - Medium, acceso: mayo 13, 2026, [https://medium.com/@heyamit10/fine-tuning-bert-for-classification-a-practical-guide-b8c1c56f252c](https://medium.com/@heyamit10/fine-tuning-bert-for-classification-a-practical-guide-b8c1c56f252c)

- Fine Tune BERT | Kaggle, acceso: mayo 13, 2026, [https://www.kaggle.com/discussions/getting-started/443332](https://www.kaggle.com/discussions/getting-started/443332)

- Modular Pipeline for Text Recognition in Early Printed Books Using Kraken and ByT5 - IRIS Unina, acceso: mayo 13, 2026, [https://www.iris.unina.it/retrieve/370597be-56d8-4e83-9354-7a5973ef45ac/Modular%20Pipeline%20for%20Text%20Recognition%20in%20Early%20Printed%20Books%20Using%20Kraken%20and%20ByT5.pdf](https://www.iris.unina.it/retrieve/370597be-56d8-4e83-9354-7a5973ef45ac/Modular%20Pipeline%20for%20Text%20Recognition%20in%20Early%20Printed%20Books%20Using%20Kraken%20and%20ByT5.pdf)

- CORAL Loss Function Overview - Emergent Mind, acceso: mayo 13, 2026, [https://www.emergentmind.com/topics/coral-loss-function](https://www.emergentmind.com/topics/coral-loss-function)

- mDAPT: Multilingual Domain Adaptive Pretraining in a Single Model - ACL Anthology, acceso: mayo 13, 2026, [https://aclanthology.org/2021.findings-emnlp.290/](https://aclanthology.org/2021.findings-emnlp.290/)

- How do I implement embedding pooling strategies (mean, max, CLS)? - Zilliz, acceso: mayo 13, 2026, [https://zilliz.com/ai-faq/how-do-i-implement-embedding-pooling-strategies-mean-max-cls](https://zilliz.com/ai-faq/how-do-i-implement-embedding-pooling-strategies-mean-max-cls)

- Beyond CLS: Advanced Pooling Strategies for Vision Transformers | by Abhishek Selokar, acceso: mayo 13, 2026, [https://medium.com/@imabhi1216/beyond-cls-advanced-pooling-strategies-for-vision-transformers-8df1785ec81c](https://medium.com/@imabhi1216/beyond-cls-advanced-pooling-strategies-for-vision-transformers-8df1785ec81c)

- dccuchile/bert-base-spanish-wwm-cased - Hugging Face, acceso: mayo 13, 2026, [https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased)

- atlijas/byt5-is-ocr-post-processing-old-texts - Hugging Face, acceso: mayo 13, 2026, [https://huggingface.co/atlijas/byt5-is-ocr-post-processing-old-texts](https://huggingface.co/atlijas/byt5-is-ocr-post-processing-old-texts)

- bertin-project/bertin-roberta-base-spanish - Hugging Face, acceso: mayo 13, 2026, [https://huggingface.co/bertin-project/bertin-roberta-base-spanish](https://huggingface.co/bertin-project/bertin-roberta-base-spanish)

- Spanish Pre-trained BERT Model and Evaluation Data - Hugging Face, acceso: mayo 13, 2026, [https://huggingface.co/papers/2308.02976](https://huggingface.co/papers/2308.02976)

- Domain-adaptative Continual Learning for Low-resource Tasks: Evaluation on Nepali - arXiv, acceso: mayo 13, 2026, [https://arxiv.org/html/2412.13860v1](https://arxiv.org/html/2412.13860v1)

- NLPAug: Python library for text data augmentation, acceso: mayo 13, 2026, [https://neurohive.io/en/novosti/nlpaug-python-library-for-text-data-augmentation/](https://neurohive.io/en/novosti/nlpaug-python-library-for-text-data-augmentation/)

- A Deep Learning Ordinal Classifier - The Science and Information (SAI) Organization, acceso: mayo 13, 2026, [https://thesai.org/Downloads/Volume16No3/Paper_30-A_Deep_Learning_Ordinal_Classifier.pdf](https://thesai.org/Downloads/Volume16No3/Paper_30-A_Deep_Learning_Ordinal_Classifier.pdf)

- [1611.05916] Squared Earth Mover's Distance-based Loss for Training Deep Neural Networks - arXiv, acceso: mayo 13, 2026, [https://arxiv.org/abs/1611.05916](https://arxiv.org/abs/1611.05916)

- NeurIPS Poster Conformal Prediction Sets for Ordinal Classification, acceso: mayo 13, 2026, [https://neurips.cc/virtual/2023/poster/71305](https://neurips.cc/virtual/2023/poster/71305)

- Fine-tuning BERT for Text classification - Kaggle, acceso: mayo 13, 2026, [https://www.kaggle.com/code/neerajmohan/fine-tuning-bert-for-text-classification](https://www.kaggle.com/code/neerajmohan/fine-tuning-bert-for-text-classification)

- Fine Tuning BERT: Text Classification - Kaggle, acceso: mayo 13, 2026, [https://www.kaggle.com/code/au1206/fine-tuning-bert-text-classification](https://www.kaggle.com/code/au1206/fine-tuning-bert-text-classification)

- Daily Papers - Hugging Face, acceso: mayo 13, 2026, [https://huggingface.co/papers?q=historic%20texts](https://huggingface.co/papers?q=historic+texts)

- This is MarIA, the first artificial intelligence in the Spanish language | datos.gob.es, acceso: mayo 13, 2026, [https://datos.gob.es/en/blog/maria-first-artificial-intelligence-spanish-language](https://datos.gob.es/en/blog/maria-first-artificial-intelligence-spanish-language)

- Linguistic corpora: the knowledge engine for AI - Datos abiertos del Gobierno de España, acceso: mayo 13, 2026, [https://datos.gob.es/en/blog/linguistic-corpora-knowledge-engine-ai](https://datos.gob.es/en/blog/linguistic-corpora-knowledge-engine-ai)

- index of the project gutenberg works of miguel de cervantes saavedra, acceso: mayo 13, 2026, [https://www.gutenberg.org/files/58328/58328-h/58328-h.htm](https://www.gutenberg.org/files/58328/58328-h/58328-h.htm)

- magistermilitum/bert_medieval_multilingual - Hugging Face, acceso: mayo 13, 2026, [https://huggingface.co/magistermilitum/bert_medieval_multilingual](https://huggingface.co/magistermilitum/bert_medieval_multilingual)

- magistermilitum/tridis_v2_HTR_historical_manuscripts - Hugging Face, acceso: mayo 13, 2026, [https://huggingface.co/magistermilitum/tridis_v2_HTR_historical_manuscripts](https://huggingface.co/magistermilitum/tridis_v2_HTR_historical_manuscripts)

- Rebels and Reformers/Cervantes - Wikisource, the free online library, acceso: mayo 13, 2026, [https://en.wikisource.org/wiki/Rebels_and_Reformers/Cervantes](https://en.wikisource.org/wiki/Rebels_and_Reformers/Cervantes)