# 🛠️ Playbook: Clasificador de Texto con Regresión Logística (NLP)

> **Qué es este documento:** la síntesis después de implementar. Integra las 4 clases en vivo (transcripciones), las notas formales del curso y el notebook real (`Clasificador_reseñas_multiclase_Regresión_logística_.ipynb`). No vuelve a explicar la teoría desde cero — para eso está la guía de pre-lectura — esto es la referencia que uso cuando vuelva a implementar algo parecido.

---

## 🗺️ El problema y el dataset

Caso de uso: clasificar reseñas de hoteles en Chicago como **verdaderas o falsas** (1600 reseñas, perfectamente balanceadas: 800/800). El dataset también tiene una segunda etiqueta de polaridad (positiva/negativa), que se usa después para construir un problema de **4 clases**.

Dos ideas del profesor que vale la pena anclar antes de tocar código:

- **No es un diccionario de palabras clave.** El modelo no parte de "estas palabras son mentira, estas son verdad" — parte de texto + etiqueta, y el modelo solito descubre qué palabras importan. Esto es justo la diferencia entre aprendizaje supervisado y un sistema de reglas.
- **Caso real de referencia: VeriPol** (España, detección de denuncias falsas de robo). Hallazgo contraintuitivo que se mencionó en clase: las denuncias con descripciones físicas muy detalladas del agresor tendían a ser **más confiables**, no menos — la hipótesis es que quien miente evita dar detalles que luego puedan usarse para acusar falsamente a alguien. Es un buen ejemplo de por qué interpretabilidad importa: el valor del modelo no es solo la predicción, es poder explicar **por qué**.

---

## 🧵 Pipeline completo (con el código real del notebook)

### 1. Preprocesamiento de texto

El orden importa y cada paso tiene una razón de negocio, no solo técnica:

```python
def elimina_stopwords(texto):
    nuevo_texto = STOPWORDS_RGX.sub('', texto)
    return nuevo_texto

def elimina_puntuacion_y_numeros(texto):
    return SOLO_LETRAS_RGX.sub(' ', texto)

def preprocesar(texto):
    texto = texto.lower()
    texto = elimina_stopwords(texto)
    texto = elimina_puntuacion_y_numeros(texto)
    texto = stemmer(texto)
    return texto
```

**Por qué este orden y no otro (de las clases):**
- Minúsculas primero, para que "Hotel" y "hotel" no sean dos tokens distintos.
- Stopwords: palabras que aparecen en casi todos los textos del idioma (artículos, preposiciones) y por eso no distinguen nada. El profesor mostró esto con un histograma de frecuencias de *Cien años de soledad*: las 20 palabras más frecuentes de cualquier libro en español son casi siempre artículos y conjunciones.
- Puntuación y números: regex `[^a-z]+` → todo lo que no sea una letra se convierte en espacio.
- Stemming vs. lematización — **no son lo mismo**, y el notebook usa solo *stemming*: el stemming es mecánico (corta terminaciones comunes como -ing, -ed) y la lematización requiere entender la gramática (enfocada en verbos en este caso). Ejemplo del notebook: `"ran run running"` → lematización da `"run run run"`, stemming da `"ran run run"` (no junta "ran" porque no detecta que es un verbo irregular). **Conclusión del profesor: la diferencia rara vez cambia mucho el desempeño del modelo final** — no vale la pena obsesionarse con elegir la técnica perfecta.

⚠️ **Decisión dependiente del contexto, no receta fija:** si trabajaras con tweets en vez de reseñas, querrías *conservar* hashtags y menciones en vez de eliminarlos como puntuación — el preprocesamiento siempre depende de qué tipo de texto tienes y para qué modelo lo vas a usar.

### 2. Vectorización: de texto a números (TF-IDF)

```python
vectorizer = TfidfVectorizer(min_df=2)
X_train = vectorizer.fit_transform(df_train['text_pp'])
```

$$
tfidf(i,x) = tf(i,x) \cdot idf(i), \qquad idf(i) = \log\left(\frac{N}{1+df(i)}\right)
$$

Lo que se explicó en clase y que **no está en las notas formales del PDF con tanto detalle**:

- **TF (frecuencia relativa)** evita que un texto largo gane artificialmente solo por ser largo. Una palabra que se repite mucho en un tweet es señal fuerte; la misma repetición en *El Quijote* no dice nada.
- **IDF castiga a las palabras que aparecen en casi todos los documentos** (conexión directa con la Ley de Zipf). El logaritmo evita que el tamaño total del corpus ($N$) sesgue la vectorización — sin el logaritmo, una base de datos gigante penalizaría distinto que una chica.
- El "+1" en el denominador es *smoothing*: evita dividir entre cero y además amortigua aún más a las palabras universales.
- **Punto clave del profesor:** empíricamente, TF-IDF casi siempre da mejores resultados que solo contar frecuencias — "está demostrado matemáticamente, no es solo que tenga más información".

⚠️ **`fit_transform` vs. `transform` — esto es para producción, no solo sintaxis:**
- `fit_transform` (en `df_train`): aprende el vocabulario Y vectoriza.
- `transform` (en `df_test` o datos nuevos): vectoriza usando el vocabulario **ya aprendido**, nunca aprende uno nuevo.
- Si llega una reseña con palabras nuevas que el vectorizador nunca vio, simplemente se ignoran — el modelo está "congelado" en las dimensiones con las que se entrenó. Si necesitas usar esas palabras nuevas, no hay otra opción que reentrenar desde cero.

### 3. Regresión logística sin penalización (línea base)

```python
clasificador_rl = LogisticRegression(penalty=None, random_state=4,
                                      max_iter=1500, tol=0.0001,
                                      solver="sag").fit(X_train, y_train)
```

Resultado típico que se vio en clase: **accuracy ≈ 100% en entrenamiento, mucho más bajo en prueba** → señal clara de sobreajuste. Dos razones de negocio (no solo "porque sí hay que regularizar"):
1. Tenemos **más columnas (palabras) que registros** en algunos escenarios — eso ya invita a sobreajuste.
2. Las palabras **no son independientes**: "room" y "service" aparecen correlacionadas. Es un modelo lineal, y la colinealidad sí le pega.

### 4. Búsqueda del mejor λ (Ridge) con validación cruzada

```python
lambdas = np.logspace(-3, 3, 25)  # o un vector más chico para ahorrar tiempo

kf = KFold(n_splits=5, shuffle=True, random_state=42)
regresion_ridge = LogisticRegression(penalty='l2', solver="sag",
                                      max_iter=1500, tol=0.0001,
                                      warm_start=True)

grid_search = GridSearchCV(estimator=regresion_ridge,
                            param_grid={'C': 1/lambdas},
                            cv=kf, return_train_score=True).fit(X_train, y_train)
```

🔑 **`C` no es `λ` — es su inverso (`C = 1/λ`).** Es la convención de scikit-learn y la causa más común de confusión: λ grande = más regularización; C grande = menos regularización.

**Analogía de las canicas (la que usó el profesor para explicar Cross-Validation):** imagina que tus reseñas son canicas verdes (verdaderas) y rojas (falsas). Con `cv=5`, separas las canicas en 5 grupos distintos. En cada una de las 5 corridas, 4 grupos entrenan y 1 evalúa — y el grupo que evalúa rota cada vez. Esto te da 5 mediciones del mismo λ con distintos datos de prueba, y promedias: así el resultado no depende de qué tan "suertudo" fue el split original.

⚠️ Una pregunta legítima que surgió en clase: *¿está mal usar todo el dataset (train + test) dentro del GridSearchCV?* No — el objetivo del CV es encontrar un **hiperparámetro robusto**, no un modelo final. El modelo final sí se entrena solo con train y se evalúa solo con test, intacto.

### 5. Coeficientes: lo que el PDF no explica con suficiente detalle

```python
coef_ridge = pd.Series(clasificador_ridge.coef_[0], index=features_names)
np.exp(coef_ridge).sort_values(ascending=False)  # importancia real
```

**El error que casi todos cometen (y que se corrigió en vivo en clase):** ver el valor crudo de $\beta_i$ y pensar que "más grande = más importante". **No es directo.** Hay que exponenciar:

$$
\frac{P(y=1|x_i+1)/P(y=0|x_i+1)}{P(y=1|x_i)/P(y=0|x_i)} = e^{\beta_i}
$$

Solo después de exponenciar se puede comparar la importancia real de una palabra para predecir una clase. En el ejemplo del notebook, "Chicago" tiene un coeficiente cercano a cero después de exponenciar — coincide con lo observado en las nubes de palabras: aunque "Chicago" es frecuente en ambas clases, casi no ayuda a distinguir entre ellas.

**Ridge vs. sin regularización (lo que se ve al graficar el histograma de coeficientes):** el rango de los coeficientes Ridge es muchísimo más angosto (ej. -7 a +4) comparado con el modelo sin penalización (ej. -30 a +20). Ridge **encoge** los coeficientes hacia cero pero rara vez los vuelve exactamente cero — eso es justo la diferencia con Lasso, que sí los fuerza a cero.

### 6. Umbral de decisión y costo de los errores

```python
predicciones['y_pred_30'] = np.where(predicciones['P(deceptive)'] > 0.3, 'deceptive', 'truthful')
```

`predict()` usa 0.5 por default — **pero ese umbral es una decisión de negocio, no matemática.** Ejemplo de clase: si vas a *publicar* reseñas y tu peor error es publicar una reseña falsa como si fuera verdadera, puedes exigir, por ejemplo, 70% de certeza antes de etiquetar algo como verdadero. Esto mueve el balance entre falsos positivos y falsos negativos sin tocar el modelo — solo cambia dónde cortas la probabilidad que ya calculó.

### 7. Curva ROC — evaluar el modelo independientemente del umbral

```python
RocCurveDisplay.from_estimator(clasificador_ridge, X_test, y_test)
```

Cada punto de la curva ROC es una matriz de confusión distinta, una por cada umbral posible. El área bajo la curva (AUC) resume **qué tan bueno es el modelo en general**, sin comprometerte todavía a un umbral específico. AUC = 1 sería el modelo perfecto (todas las matrices de confusión te hacen feliz); AUC = 0.5 es equivalente a adivinar.

### 8. Extensión a multiclase

```python
y_train = df_train['deceptive'].str.cat(df_train['polarity'], sep='-')
# resultado: 4 clases -> deceptive-positive, deceptive-negative, truthful-positive, truthful-negative

clasificador_multiclase = LogisticRegression(penalty=None, solver="sag",
                                              max_iter=1500, tol=0.0001,
                                              random_state=4).fit(X_train, y_train)
```

`sklearn` resuelve esto internamente como **uno contra todos (one-vs-rest)**: entrena una regresión logística binaria por cada clase contra el resto. Esto es importante para los parámetros: si tu problema es multiclase, `solver="newton-cg"` puede no estar disponible — hay que revisar la tabla de compatibilidad solver/penalización antes de correr el GridSearch.

---

## 🎛️ Parámetros clave — tabla de referencia rápida

| Parámetro | Qué controla | Gotcha visto en clase |
|---|---|---|
| `C` | Inverso de la fuerza de regularización | `C = 1/λ`; C grande = poca regularización |
| `penalty` | Tipo de penalización (`l1`, `l2`, `None`) | No todos los `solver` soportan todas las penalizaciones — revisar tabla de sklearn |
| `solver` | Algoritmo de optimización | `sag` se parece al descenso de gradiente del perceptrón; `newton-cg` no soporta multiclase en todos los casos |
| `max_iter` | Tope de iteraciones | Si sale el warning de no convergencia, primero sube esto antes de cambiar `tol` |
| `tol` | Criterio de paro (qué tan chico debe ser el cambio entre iteraciones para detenerse) | Subir `tol` para forzar convergencia más rápido puede generar un modelo subajustado |
| `warm_start` | Reutiliza coeficientes de la corrida anterior | Acelera mucho el GridSearchCV cuando se prueban muchos λ seguidos |
| `n_jobs` | Paraleliza el entrenamiento | Solo aplica de forma útil en problemas multiclase (uno contra todos) |

---

## 🧠 Por qué regresión logística (y no perceptrón o árbol)

Idea central de la clase 2, vale la pena tenerla siempre presente para justificar la elección de modelo (literalmente lo que pide el examen final del módulo):

```
Árbol de decisión          → muy interpretable, precisión media
        ↓
Regresión logística (+Ridge) → punto intermedio: interpretable Y precisa
        ↓
Perceptrón                 → muy preciso, nada interpretable
```

La regresión logística es atractiva precisamente porque no obliga a elegir entre explicar el modelo o que funcione bien. Es lineal (no captura no-linealidades como un árbol), pero a cambio cada coeficiente tiene una lectura directa vía $e^{\beta_i}$.

---

## ⚠️ Errores y advertencias reales que aparecieron en clase

- **`ConvergenceWarning` por `max_iter` insuficiente:** primero subir iteraciones, luego (si no funciona) ajustar `tol`. El profesor probó 500 → 1000 → ajustó `tol` a 0.002 → volvió a bajar iteraciones a 15 y sí convergió. No hay una receta fija, es prueba y error documentado.
- **R² no aplica a un clasificador.** Surgió la duda de por qué `LogisticRegression.score()` no da error pero calcular R² manualmente sí — R² es una métrica de regresión, no de clasificación, aunque el nombre "regresión logística" invite a la confusión.
- **Coeficientes en cero "inesperados" con Ridge:** en vivo, el profesor encontró ~24 coeficientes exactamente en cero con penalización Ridge, cuando en teoría Ridge no debería llevar nada exactamente a cero (eso es trabajo de Lasso). Quedó como algo a revisar — útil recordar que **Ridge en la práctica con un solver numérico puede producir coeficientes muy cercanos a cero que se redondean a cero**, no es necesariamente un error de código.
- **No metas `solver` y `penalty` juntos al GridSearchCV "a fuerza bruta".** No todas las combinaciones son válidas, y es computacionalmente muy caro probar todas. Mejor: fija la penalización que ya sabes que necesitas (aquí, L2 por correlación entre palabras) y busca solo sobre `C`.

---

## 🏗️ Checklist para adaptar esto a un problema propio

- [ ] ¿Mi variable de salida es realmente binaria o categórica (no ordenada)? Si es continua, esto no aplica — es forecasting, no clasificación.
- [ ] ¿Tengo una etiqueta histórica confiable? Sin supervisión no hay modelo, sin importar qué tan bueno sea el algoritmo — la base de datos importa más que el algoritmo.
- [ ] ¿El preprocesamiento de texto tiene sentido para mi tipo de fuente? (reseñas ≠ tickets de soporte ≠ tweets — revisar qué conservar y qué tratar como stopword según el contexto)
- [ ] ¿Ya decidí si necesito interpretabilidad (explicarle a alguien por qué el modelo decidió algo) o solo precisión cruda?
- [ ] ¿Definí cuál es el error más costoso para el negocio antes de fijar el umbral de decisión, en vez de usar 0.5 por default?
- [ ] ¿Validé el hiperparámetro de regularización con K-Fold en vez de un solo split de train/test?

---

## 🧾 Glosario rápido (términos que aparecieron en clase y no estaban en el PDF)

| Término | Significado en este contexto |
|---|---|
| Token | Unidad mínima en la que se divide el texto (aquí, palabras o raíces de palabras) |
| Corpus | El conjunto completo de textos con el que se trabaja |
| Vocabulario | Conjunto de tokens únicos que se convierten en columnas del vector |
| Stopwords | Palabras muy frecuentes en el idioma que no ayudan a distinguir textos |
| Stemming | Normalización mecánica (corta terminaciones comunes) |
| Lematización | Normalización lingüística (requiere identificar la categoría gramatical) |
| Ley de Zipf | Relación entre el ranking de frecuencia de una palabra y su importancia relativa — base del IDF |
| `fit_transform` vs `transform` | Aprender vocabulario + vectorizar vs. solo vectorizar con vocabulario ya aprendido |
| AUC | Área bajo la curva ROC; mide la calidad del modelo independientemente del umbral elegido |
| One-vs-rest | Estrategia para resolver problemas multiclase entrenando un clasificador binario por clase |
