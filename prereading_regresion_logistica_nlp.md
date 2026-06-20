# 📖 Guía de Pre-lectura: Regresión Logística, Vectorización de Texto y Regularización

> **¿Para qué sirve esta guía?**
> Antes de leer las notas del curso, esta guía te da el vocabulario y los conceptos base
> para que la lectura fluya. No necesitas memorizarla — léela una vez y úsala como referencia.

---

## 🗺️ Mapa del Tema

Estas notas resuelven un problema muy distinto al de la regresión que has estado viendo (forecasting continuo): aquí la pregunta no es "¿cuánto?" sino "¿cuál de dos clases?". El hilo conductor es la **regresión logística**, un algoritmo de clasificación binaria que, a diferencia del perceptrón, no solo dice "clase A" o "clase B" sino que asigna una *probabilidad* a cada clase. El caso de uso estrella del documento es clasificar texto (reseñas de hoteles auténticas vs. falsas), por lo que antes de llegar al modelo necesitas entender cómo un texto se convierte en un vector de números (Bag of Words, TF-IDF). Al final, el documento extiende la idea a más de dos clases (softmax) y resuelve el problema de sobreajuste con dos técnicas de regularización: Ridge y Lasso.

---

## 🔤 Notación que Vas a Ver

| Símbolo | Significado | Ejemplo |
|---|---|---|
| $\mathbb{R}^d$ | Espacio de vectores de tamaño $d$ con entradas reales | $\mathbf{x} \in \mathbb{R}^d$ |
| $\langle \mathbf{x}, \mathbf{y} \rangle$ | Producto punto entre dos vectores | $\langle \mathbf{x}, \mathbf{y} \rangle = x_1y_1 + \dots + x_dy_d$ |
| $\sigma(x)$ | Función sigmoide | $\sigma(x) = \dfrac{1}{1+e^{-x}}$ |
| $\boldsymbol{\beta}$ | Vector de pesos del modelo (coeficientes) | $\boldsymbol{\beta} \in \mathbb{R}^d$ |
| $err_{Log,\beta}$ | Función de error/pérdida de la regresión logística | ver sección de Objetos Matemáticos |
| $tf(i)$, $idf(i)$ | Frecuencia de término / frecuencia inversa de documento | usados para vectorizar texto |
| $\|\boldsymbol{\beta}\|_0, \|\boldsymbol{\beta}\|_1, \|\boldsymbol{\beta}\|_2$ | Normas 0, 1 y 2 de un vector | ver sección de normas |
| $S = \{(x_i, y_i)\}$ | Conjunto de entrenamiento (datos + etiquetas) | $y_i \in \{0,1\}$ |

---

## 🧱 Conceptos Base

> Estos son los bloques de construcción. Si ya los conoces, repasa rápido. Si no, léelos con calma.

### Probabilidad condicional y Teorema de Bayes

La probabilidad condicional responde "¿qué tan probable es A, sabiendo que ya ocurrió B?". El Teorema de Bayes es la herramienta que permite "invertir" esa pregunta: si sabes $P(B|A)$, puedes calcular $P(A|B)$. Es la pieza central que conecta "qué tan probable es que esta palabra aparezca dado que la reseña es falsa" con "qué tan probable es que la reseña sea falsa dado que apareció esta palabra" — que es justo lo que queremos predecir.

$$
P(A|B) = \frac{P(A \cap B)}{P(B)}
$$

**¿Por qué importa?** Toda la justificación probabilística de la regresión logística (por qué tiene sentido usar la función sigmoide) se construye aplicando Bayes y despejando.

### Logaritmo y exponencial como pareja inversa

El logaritmo convierte productos en sumas ($\log(\alpha\gamma) = \log(\alpha) + \log(\gamma)$) y cocientes en restas. Esta propiedad es la que permite pasar de "razón de probabilidades" (un cociente, difícil de modelar) a una suma lineal de variables (fácil de modelar con $\boldsymbol{\beta} \cdot \mathbf{x}$).

**¿Por qué importa?** Es el truco algebraico detrás de la fórmula que conecta a la regresión logística con una ecuación lineal: el log-odds.

### Vectorización de texto (Bag of Words y TF-IDF)

Un modelo de machine learning no puede "leer" texto, necesita números. La vectorización asigna a cada texto un vector $\mathbf{x} \in \mathbb{R}^d$ donde $d$ es el tamaño del vocabulario, y cada coordenada $x_i$ mide qué tan presente está la palabra $i$. Hay varias formas de definir ese valor: frecuencia simple, frecuencia relativa a la longitud del texto ($tf$), frecuencia en todo el corpus ($df$), y una combinación que penaliza palabras muy comunes en todo el corpus ($tf \cdot idf$).

**¿Por qué importa?** Es el paso de preprocesamiento obligatorio antes de poder aplicar cualquier regresión logística a un problema de texto.

---

## 📐 Objetos Matemáticos Clave

> Aquí viven las "cosas" con las que vas a trabajar: vectores, matrices, funciones, distribuciones.

### 📦 Función sigmoide — Función

**Qué es:** Una función que "aplasta" cualquier número real al intervalo $(0,1)$, lo que la hace perfecta para representar una probabilidad.

**Forma general:**
$$
\sigma(x) = \frac{1}{1+e^{-x}}
$$

**Propiedades que vas a usar:**
- Cuando $x$ es muy grande (positivo o negativo), $\sigma(x)$ se acerca a 1 o a 0.
- Cuando $x \approx 0$, $\sigma(x) \approx 0.5$ (máxima incertidumbre).
- Es la solución a la ecuación diferencial logística de Verhulst (crecimiento poblacional con límite de recursos) — de ahí el nombre "regresión logística".

### 📦 Función de pérdida logística — Función

**Qué es:** Mide qué tan "mal" predijo el modelo una etiqueta real, comparando la probabilidad predicha contra la etiqueta verdadera (0 o 1).

**Forma general:**
$$
err_{Log,\beta}(x,y) = y\cdot\log\left(\frac{1}{1+e^{-x\cdot\beta}}\right) + (1-y)\log\left(1-\frac{1}{1+e^{-x\cdot\beta}}\right)
$$

**Propiedades que vas a usar:**
- El objetivo del entrenamiento es minimizar el promedio de esta pérdida sobre todos los ejemplos.
- No tiene solución cerrada exacta: se resuelve con descenso de gradiente o método de Newton.

### 📦 Normas en $\mathbb{R}^d$ — Objeto algebraico

**Qué es:** Distintas formas de medir el "tamaño" de un vector de coeficientes $\boldsymbol{\beta}$. Cada norma penaliza de forma distinta a los coeficientes grandes.

**Forma general:**
$$
\|\boldsymbol{\beta}\|_2 = \sqrt{\sum_i \beta_i^2}, \qquad \|\boldsymbol{\beta}\|_1 = \sum_i |\beta_i|, \qquad \|\boldsymbol{\beta}\|_0 = \sum_i \mathbb{1}_{\beta_i \neq 0}
$$

**Propiedades que vas a usar:**
- La norma 0 cuenta cuántos coeficientes son distintos de cero (ideal para "modelos dispersos", pero no es convexa: imposible de optimizar directamente).
- La norma 1 (Lasso) es una aproximación convexa de la norma 0.
- La norma 2 al cuadrado (Ridge) penaliza coeficientes grandes sin forzarlos a cero.

---

## 📜 Resultados Importantes

> Estos son los teoremas y proposiciones que vas a encontrar. No tienes que probarlos ahora,
> solo entender qué dicen y por qué son útiles.

### 🏛️ Ley de Probabilidad Total

**En palabras:** Si divides todo el espacio en partes que no se traslapan (una partición), la probabilidad de cualquier evento se puede calcular sumando sus probabilidades condicionales a cada parte, ponderadas por la probabilidad de esa parte.

**Formalmente:**
$$
P(B) = \sum_{i=1}^{n} P(B|A_i)\cdot P(A_i)
$$

**Condiciones:** $A_1,\dots,A_n$ deben formar una partición del espacio.
**Utilidad:** Es el paso intermedio que permite derivar, junto con Bayes, la fórmula $P(Y=1|X) = \sigma(\cdot)$.

### 🏛️ El estimador logístico coincide con el de máxima verosimilitud

**En palabras:** Si los datos cumplen ciertas hipótesis (errores independientes e idénticamente distribuidos), minimizar la función de pérdida logística es matemáticamente equivalente a encontrar los parámetros más "creíbles" dado lo que observaste.

**Condiciones:** Separabilidad lineal y logística (existe un $\boldsymbol{\beta}^*$ verdadero más un error aleatorio).
**Utilidad:** Justifica por qué optimizar esa pérdida específica (y no otra) tiene sentido estadístico.

### 🏛️ Justificación de Lasso vía compressed sensing

**En palabras:** Si el verdadero vector de coeficientes tiene pocos valores distintos de cero (es "disperso") y tienes suficientes datos, Lasso recupera exactamente esos coeficientes con alta probabilidad.

**Utilidad:** Es la razón teórica para preferir la norma 1 sobre la norma 0 cuando quieres un modelo con pocas variables relevantes.

---

## ⚠️ Puntos de Confusión Frecuentes

- **Etiquetas $\{0,1\}$ en vez de $\{-1,+1\}$:** A diferencia del perceptrón, aquí las etiquetas son 0 y 1 porque se interpretan como una probabilidad, no como una dirección. El valor numérico en sí no tiene significado especial, solo distingue las dos clases.
- **"Regresión" que en realidad clasifica:** El nombre es engañoso. El modelo predice una probabilidad continua, pero se usa para tomar una decisión binaria (clasificar). Lo que sí es lineal es el *log-odds* (cociente de probabilidades en escala logarítmica), no la probabilidad misma.
- **Norma 0 vs. norma 1:** La intuición de Lasso es la norma 0 (forzar coeficientes a cero exactamente), pero como esa norma no es convexa y no se puede optimizar, se usa la norma 1 como sustituto que sí funciona en la práctica.
- **Interpretar $\beta_i$ directamente:** El valor de $\beta_i$ no se interpreta igual que en regresión lineal. Hay que exponenciarlo ($e^{\beta_i}$) para obtener un "razón de momios" (odds ratio): cuánto se multiplica la razón de probabilidades al aumentar esa variable en una unidad.

---

## 🔗 Conexiones con lo que Ya Sabes

- Si viste **el perceptrón**: la regresión logística es prácticamente el mismo esquema (un hiperplano que separa clases) pero cambiando la función de activación de un escalón a una sigmoide, lo que te da probabilidades en vez de una decisión dura.
- Si viste **árboles de decisión**: el documento los menciona como una alternativa para resolver clasificación multiclase, comparándolos con los enfoques One vs. One / One vs. All basados en modelos binarios.
- Si ya usas **regresión lineal**: la lógica de interpretar coeficientes por signo y magnitud es similar, pero aquí necesitas un paso extra (exponenciar) por la no linealidad de la sigmoide.

---

## 🛠️ Aplicación Práctica

### ¿Cuándo usarías este modelo?

| Situación | ¿Lo usarías? |
|---|---|
| Clasificar texto en dos categorías (ej. spam vs. no spam, reseña falsa vs. auténtica) | ✅ Es el caso de uso central del documento |
| Detectar fraude en transacciones (datos muy desbalanceados) | ✅ El documento señala explícitamente que el desbalance no afecta la precisión del modelo, solo requiere ajustar $\beta_0$ |
| Identificar si un SKU está en stockout a partir de variables de órdenes de compra | ✅ Es un problema de clasificación binaria (stockout / no stockout) — encaja directamente con este marco si decides modelarlo así en vez de con reglas fijas |
| Pronosticar el volumen de ventas semanal por SKU | ❌ Es una variable continua (regresión, no clasificación); este modelo no aplica directamente |
| Predecir entre más de dos categorías (ej. tipo de producto entre 5 opciones) | ⚠️ Solo con la extensión softmax/multinomial de la sección de clasificación multiclase |

### Jerarquía o evolución del modelo

```
Perceptrón (clasificación binaria, decisión dura, sin probabilidad)
        ↓
Regresión logística (clasificación binaria con probabilidad, función sigmoide)
        ↓
Regresión softmax / multinomial (clasificación multiclase, generaliza la sigmoide)
        ↓
Redes neuronales profundas (la capa final usa softmax para clasificación)
```

### 🔍 Qué observar cuando leas el código del curso

- Cómo se construye la matriz Bag of Words (1600 documentos × 3,528 palabras) y si se usa frecuencia simple o TF-IDF.
- Qué preprocesamiento de texto se aplica antes de vectorizar (stopwords, minúsculas, stemming/lematización).
- Cómo se codifican las etiquetas (reseña auténtica = 0/1, falsa = el otro valor).
- Si se aplica la penalización Ridge y qué valor de $\lambda$ se elige.
- Qué métrica se reporta para evaluar el modelo, considerando que el dataset está balanceado (800 vs. 800).

---

## ✅ Checklist Antes de Leer las Notas

Antes de abrir el PDF del curso, asegúrate de poder responder:

- [ ] ¿Qué hace la función sigmoide y por qué es útil para representar probabilidades?
- [ ] ¿Cómo se convierte un texto en un vector numérico (Bag of Words / TF-IDF)?
- [ ] ¿Qué dice la Ley de Probabilidad Total en tus propias palabras?
- [ ] ¿Cuál es la diferencia práctica entre la penalización Ridge (norma 2) y Lasso (norma 1)?
- [ ] ¿Cómo se interpreta el coeficiente $\beta_i$ de una variable en la predicción final?
