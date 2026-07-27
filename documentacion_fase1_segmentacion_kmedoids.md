# Documentación Técnica — Fase 1: Segmentación de Comercios mediante K-Medoids

**Proyecto:** Prueba Técnica Pragma — Analítica de Banca Digital  
**Objetivo corporativo:** Incrementar en un 50% las comisiones por uso de la API de pago para el año 2022  
**Fecha de corte de datos:** Mayo 2021  
**Versión del documento:** 1.0  

---

## 1. Descripción Integral del Flujo ETL

### 1.1 Ingesta de Archivos Crudos

El pipeline ETL (`src/etl.py`) ingiere dos fuentes de datos en formato CSV con separador punto y coma (`;`) y codificación `latin-1`:

| Archivo | Contenido | Registros |
|---------|-----------|-----------|
| `DataTransacciones_PruebasIngreso_Comercios.csv` | Catálogo de comercios con información de vinculación, terminales, categoría, ubicación y montos de ventas | 3,277 filas |
| `DataTransacciones_PruebasIngreso_Transacciones.csv` | Historial transaccional mensual por comercio (`mes_anio`, `cantidad_transacciones`) | Múltiples períodos |

Ambos archivos se cargan en modo `dtype=str` para garantizar control total sobre la limpieza y evitar coerciones implícitas de tipos. Los nombres de columna se normalizan a `snake_case` mediante la función `_normalize_column_names()`, que convierte a minúsculas, reemplaza espacios, guiones y barras por guiones bajos, colapsa múltiples guiones bajos consecutivos y elimina guiones bajos al inicio y final.

### 1.2 Limpieza, Normalización, Imputación y Validación

#### 1.2.1 Parseo de Fechas en Español

Las fechas en el dataset de comercios utilizan un formato no estándar en español abreviado:

```
"mar. 26- 2021 12:40pm"
```

La función `_parse_spanish_date()` aplica una expresión regular para extraer los componentes: abreviatura del mes, día, año, hora, minuto y meridiano (am/pm). Se utiliza un diccionario de mapeo de meses (`ene` → 1, `feb` → 2, …, `dic` → 12). La conversión de formato 12h a 24h se realiza según las reglas estándar (12pm = 12, 1pm–11pm = +12, 12am = 0). Se parsean tres columnas de fecha:

- `fecha_vinculacion_datafono` → `fecha_vinculacion`
- `fecha_de_creacion_comercio` → `fecha_creacion`
- `fecha_1ra_transaccion` → `fecha_primera_transaccion`

Las fechas se convierten al tipo nativo `datetime` de pandas tras el parseo.

#### 1.2.2 Conversión de Montos Colombianos

Los montos en los archivos fuente utilizan el formato numérico colombiano con punto como separador de miles (ej. `"6.542.000"`). La función `_clean_amount()` elimina los puntos (separador de miles) y convierte la coma decimal a punto para obtener un `float` estándar. Se procesan dos columnas monetarias:

- `monto_ventas_acumulado` — ventas acumuladas históricas del comercio
- `monto_ventas_ultimo_mes` — ventas del último mes registrado

#### 1.2.3 Normalización de Texto

Las columnas categóricas (`ciudad`, `categoria_comercio`, `sub_categoria_comercio`) se normalizan aplicando `strip()` y colapsando espacios múltiples (`\s+` → espacio simple), eliminando variaciones causadas por espacios sobrantes en los datos fuente.

#### 1.2.4 Validaciones de Calidad

Durante la carga se ejecutan validaciones automáticas:

- **Duplicados en `id_comercio`:** se detectan y reportan, pero no se eliminan — representan múltiples terminales (datafónos) asociados al mismo comercio, y se consolidan en una etapa posterior.
- **Duplicados en transacciones:** combinaciones repetidas de `(id_comercio, mes_anio_transaccion)`.
- **Nulos en claves primarias:** se reportan como advertencias (`WARN`).
- **Filtrado de período:** las transacciones se acotan a los últimos 6 meses tomando como referencia la fecha máxima presente en el dataset (`max(fecha_transaccion) - 6 meses`).

#### 1.2.5 Conversión de `mes_anio` en Transacciones

La columna `mes_anio_transaccion` (formato `"3-2021"`) se convierte a `datetime` representando el primer día del mes correspondiente mediante `pd.Timestamp(year, month, day=1)`. La cantidad de transacciones se convierte a entero con `pd.to_numeric(errors="coerce")`, imputando valores no numéricos con `0`.

### 1.3 Reglas de Negocio en la Consolidación por `id_comercio`

La función `consolidate_comercios()` agrega múltiples terminales por comercio en una única fila representativa. Las reglas de agregación responden a la semántica del negocio:

| Campo | Regla de Agregación | Justificación |
|-------|---------------------|---------------|
| `fecha_vinculacion`, `fecha_creacion`, `fecha_primera_transaccion` | `min` (fecha más antigua) | Refleja el momento real en que el comercio inició su relación con la plataforma |
| `monto_ventas_acumulado`, `monto_ventas_ultimo_mes` | `sum` (suma de todos los terminales) | El volumen total de ventas del comercio es la suma de lo que transa por cada datafono |
| `categoria_comercio`, `sub_categoria_comercio`, `ciudad` | `moda` (valor más frecuente) | La categoría predominante entre los terminales representa al comercio; en caso de empate, se toma el primer valor |
| `id_terminal` | Lista de terminales únicos | Trazabilidad y auditoría |

La consolidación produce además columnas de auditoría:

- `n_terminals`: cantidad de terminales únicos asociados al comercio.
- `duplicated_flag`: `True` si el comercio tiene más de un terminal (detecta comercios multi-datafono).
- `registros_origen`: lista de índices del DataFrame original (trazabilidad completa).

### 1.4 Construcción de Métricas RFM y Derivadas

La función `generate_rfm()` calcula las métricas fundamentales a nivel de comercio:

#### Recencia (R) — `recency_dias`

Días transcurridos desde la última transacción del comercio hasta la fecha de corte del análisis (`max(fecha_transaccion)` del dataset de transacciones). **No se utiliza la fecha actual del sistema**, garantizando reproducibilidad. El cálculo es:

```
recencia_dias = max_date - max(fecha_transaccion por comercio)
```

Valores bajos indican actividad transaccional reciente; valores altos señalan riesgo de abandono. La recencia tiene interpretación lineal directa: cada día adicional sin transaccionar incrementa proporcionalmente el riesgo de churn. Por esta razón, no se le aplica transformación logarítmica en etapas posteriores.

#### Frecuencia (F) — `frecuencia`

Suma total de transacciones del comercio en la ventana analizada (últimos 6 meses). Se calcula como:

```
frecuencia = sum(cantidad_transacciones) agrupado por id_comercio
```

#### Monetary (M) — `monetary`

Monto acumulado de ventas consolidado de todos los terminales del comercio. **Fuente:** `monto_ventas_acumulado` del dataset de comercios (ya consolidado en la etapa anterior), no del dataset de transacciones — este último solo contiene conteos de transacciones, no montos transados.

#### Métricas Derivadas

| Métrica | Fórmula | Interpretación |
|---------|---------|----------------|
| `ticket_promedio` | `monetary / frecuencia` | Valor promedio por transacción. Tipifica el tipo de negocio: micro-transacciones vs. ventas de alto valor unitario |
| `comision_generada` | `monetary * 0.05` | Comisión del 5% que el banco recibe por cada transacción. Es la variable directamente vinculada al objetivo de negocio (+50% en comisiones) |
| `actividad_mensual_promedio` | `frecuencia / meses_activos` | Intensidad transaccional mensual. Un comercio con 60 transacciones en 3 meses activos tiene actividad promedio de 20 tx/mes |
| `meses_activos` | `nunique(fecha_transaccion)` | Número de meses distintos con al menos una transacción registrada (máximo 6 en la ventana de análisis) |

Los valores `NaN` en `ticket_promedio` y `actividad_mensual_promedio` se producen cuando `frecuencia = 0` o `meses_activos = 0` respectivamente (comercios sin transacciones en el período). Todas las métricas monetarias se redondean a 2 decimales.

### 1.5 Justificación Técnica de Cada Transformación

| Transformación | Justificación Técnica |
|----------------|----------------------|
| Parseo de fechas con regex + diccionario de meses | El formato `"mar. 26- 2021 12:40pm"` no es reconocido por `pd.to_datetime` ni por `dateutil`. La expresión regular captura de forma robusta las variaciones de espaciado en los datos fuente. El diccionario explícito de meses evita dependencia de configuraciones regionales del sistema operativo |
| Limpieza de montos con reemplazo de separadores | El formato numérico colombiano usa `.` como separador de miles y no es interpretable directamente como float. La estrategia `replace(".", "").replace(",", ".")` convierte `"6.542.000"` → `"6542000"` → `6542000.0` |
| `left join` comercios ← RFM | Conserva todos los comercios del catálogo, incluso aquellos sin transacciones en el período analizado. Esto es crítico para identificar comercios inactivos (potencial churn) |
| Relleno de métricas nulas con `0` | Comercios sin transacciones reciben `frecuencia=0`, `recencia_dias=0`, `meses_activos=0`. Esto permite que el modelo de segmentación los clasifique correctamente como inactivos en lugar de excluirlos |
| Filtrado de 6 meses con fecha máxima del dataset | La fecha de corte se calcula como `max(fecha_transaccion) - 6 meses + 1 día`. Usar la fecha máxima del dataset en lugar de la fecha actual del sistema hace el análisis determinista y reproducible en cualquier momento futuro |
| `dtype=str` en carga inicial | Evita que pandas infiera tipos incorrectos (ej. interpretar un ID como entero y perder ceros a la izquierda, o parsear mal fechas en formato no estándar). Todo el tipado se realiza de forma controlada en etapas posteriores |

### 1.6 Exportación del Dataset Final

El pipeline exporta `comercios_segmented_prep.csv` en `data/processed/` con las siguientes características:

- **Formato:** CSV, UTF-8, sin índice
- **Registros:** 3,015 comercios (una fila por `id_comercio`)
- **Columnas:** información del comercio (categoría, ciudad, fechas, terminales) + métricas RFM calculadas + KPIs derivados
- **Trazabilidad:** columnas `id_terminals`, `n_terminals`, `duplicated_flag`, `registros_origen` preservan la auditoría completa

---

## 2. Preparación de Datos para Modelado

### 2.1 Selección del Feature Set RFM y Justificación Analítica

Se definieron dos conjuntos de features candidatos:

| Feature Set | Variables | Cardinalidad |
|-------------|-----------|-------------|
| **Set A (RFM Clásico)** — *activo* | `recency_days`, `frequency`, `monetary` | 3 dimensiones |
| **Set B (RFM Extendido)** | Set A + `ticket_avg`, `n_terminals` | 5 dimensiones |

**Justificación de la selección del Set A como activo:**

- El marco RFM clásico (Recencia, Frecuencia, Valor Monetario) es el estándar consolidado en la industria financiera para segmentación de clientes. Sus tres dimensiones capturan los ejes fundamentales del comportamiento transaccional: ¿hace cuánto transaccionó? (urgencia), ¿con qué frecuencia lo hace? (engagement), ¿cuánto valor genera? (tamaño).
- `ticket_avg` es una variable derivada (`monetary / frequency`) que introduce multicolinealidad potencial con `monetary` y `frequency`. Si bien aporta información sobre el tipo de negocio, se prefiere mantener el espacio de features minimalista y ortogonal.
- `n_terminals` es una variable casi constante: más del 98% de los comercios tiene exactamente 1 terminal. Variables con varianza cercana a cero no contribuyen a la separación de clusters y añaden ruido dimensional.

### 2.2 Análisis de Correlación y Descarte por Multicolinealidad

Se calculó la matriz de correlación de Pearson para ambos feature sets. El criterio de descarte establece que si dos features presentan `|r| > 0.85`, se retiene la más directamente observable:

- En el **Set A** (3 variables RFM), las correlaciones entre `recency_days`, `frequency` y `monetary` se mantienen por debajo del umbral de 0.85, confirmando que las tres dimensiones capturan información complementaria y no redundante.
- En el **Set B**, `monetary` y `ticket_avg` presentan correlación elevada (derivación algebraica: `ticket_avg = monetary / frequency`). Dado que `monetary` es directamente observable (suma de ventas) mientras que `ticket_avg` es una variable derivada, se prefiere `monetary` en caso de descarte.

El Set A no activa el criterio de descarte por multicolinealidad, validando su uso como espacio de features para el modelado.

### 2.3 Análisis de Distribuciones y Asimetría — Justificación de `log1p`

Las variables `frequency` y `monetary` en datasets financieros presentan típicamente distribuciones con cola derecha larga (*power-law* o *log-normal*): la mayoría de los comercios tienen pocas transacciones y montos moderados, mientras que una minoría concentra volúmenes extremadamente altos.

El análisis de asimetría (*skewness*) sobre los datos crudos confirma esta hipótesis:

| Variable | Skewness | ¿Transformar? | Justificación |
|----------|----------|---------------|---------------|
| `frequency` | > 3.0 | **Sí (`log1p`)** | Cola financiera muy larga (max observado = 740 transacciones en 6 meses). Sin transformación, K-Medoids asignaría pesos desproporcionados a los comercios de altísima frecuencia |
| `monetary` | > 3.0 | **Sí (`log1p`)** | Distribución log-normal típica en finanzas (max ≈ COP $217 M). La transformación logarítmica linealiza la escala, haciendo que diferencias relativas —no absolutas— definan los clusters |
| `recency_days` | < 3.0 | **No** | La recencia tiene interpretación lineal directa en el contexto de riesgo de churn: 60 días de inactividad representan exactamente el doble de riesgo que 30 días. Aplicar `log1p` destruiría esta interpretación de negocio |

**Uso de `log1p` en lugar de `log`:** `np.log1p(x) = ln(x + 1)` es numéricamente estable para valores cercanos a cero y maneja correctamente `x = 0` (caso de comercios con `frequency = 0` o `monetary = 0`) sin producir `-inf`. Esto es particularmente relevante para el dataset, donde existen comercios sin transacciones en el período.

### 2.4 Justificación Técnica del Escalador — RobustScaler vs. StandardScaler

El dataset contiene valores extremos (*outliers*) que afectan de manera diferenciada a cada método de escalado:

| Variable | Outliers observados | Impacto en StandardScaler |
|----------|---------------------|---------------------------|
| `monetary` | Máx ≈ COP $217 M (percentil 99.9 ≈ COP $52 M) | La media y la desviación estándar se inflan, comprimiendo el 99% de los datos cerca de 0 |
| `ticket_avg` | Máx ≈ COP $29 M | Distorsiona la escala de todo el feature space |
| `commission` | Hereda los outliers de `monetary` (relación lineal vía `commission = monetary * 0.05`) | Mismo efecto de compresión |

**Se selecciona `RobustScaler`** por las siguientes razones:

1. **Insensibilidad a outliers:** `RobustScaler` utiliza la mediana (Q2) como medida de tendencia central y el rango intercuartílico (IQR = Q3 − Q1) como medida de dispersión. Ambos son estadísticos robustos que no se ven afectados por valores extremos. La transformación es `(x − Q2) / IQR` en lugar de `(x − μ) / σ`.

2. **Preservación de la estructura real de los datos:** El 99% de los comercios reciben una representación fiel en el espacio de features escalado. Los clusters quedan definidos por diferencias estructurales reales en comportamiento RFM, no por la presencia o ausencia de un outlier.

3. **Sinergia con K-Medoids:** K-Medoids ya es intrínsecamente robusto a outliers (el medoide es un punto real, no un promedio). La combinación **RobustScaler + K-Medoids** ofrece doble protección: el escalado no distorsiona las distancias relativas, y el algoritmo no es sensible a puntos extremos.

### 2.5 Argumentación sobre Mantener Outliers

Los comercios con valores extremos en `monetary` (top 0.1%, con ventas acumuladas superiores al percentil 99.9) **se mantienen íntegramente en la muestra**. Esta decisión se fundamenta en tres pilares:

1. **K-Medoids es resistente a outliers:** A diferencia de K-Means —donde un outlier puede "arrastrar" el centroide y distorsionar la asignación de clusters enteros—, K-Medoids selecciona puntos reales como representantes. Un comercio con `monetary` extremo será asignado al cluster cuyo medoide sea más cercano en el espacio RFM, sin distorsionar la posición de dicho medoide.

2. **Los outliers son comportamiento real del negocio:** Un comercio que transa COP $217 M en 6 meses no es un error de medición sino un cliente genuinamente grande. Excluirlo sería perder información valiosa sobre el segmento de mayor aporte a las comisiones.

3. **RobustScaler mitiga el impacto en la escala:** Aunque los outliers se mantienen, `RobustScaler` asegura que no dominen la escala de las variables, ya que utiliza la mediana y el IQR —insensibles a valores extremos— como parámetros de normalización.

---

## 3. Modelado con K-Medoids (PAM)

### 3.1 Justificación Técnica: K-Medoids vs. K-Means

La elección de K-Medoids (Particionamiento Alrededor de Medoides, PAM) sobre K-Means para este problema se sustenta en cuatro criterios fundamentales:

| Criterio | K-Means | K-Medoids (PAM) | Relevancia para el caso |
|----------|---------|-----------------|------------------------|
| **Resistencia a outliers** | Baja — la media (centroide) es sensible a valores extremos; un solo outlier puede desplazar el centroide fuera de la nube de puntos reales | Alta — el medoide es siempre un punto real de la muestra; los outliers no afectan su posición | Crucial: el dataset contiene comercios con `monetary` hasta COP $217 M y `frequency` hasta 740 tx, que son outliers genuinos del negocio |
| **Interpretabilidad** | El centroide es una construcción matemática abstracta (promedio de coordenadas) sin correspondencia con ningún comercio real | Cada cluster está representado por un `id_comercio` real y trazable | Fundamental para la adopción comercial: el equipo de negocio puede "visitar" al comercio medoide y entender qué tipifica a cada segmento |
| **Auditabilidad** | Difícil de auditar: ¿qué significa un centroide con `log1p(monetary) = 14.2`? | Fácil de auditar: "El comercio representativo del segmento Champions es el ID 96FVKJJ572, que transó $X en el período" | Requisito implícito de la prueba: los resultados deben ser explicables al cliente |
| **Métrica de distancia** | Distancia euclidiana al centroide | Distancia euclidiana al medoide (PAM minimiza la suma de distancias intra-cluster) | Equivalente en el espacio de features tras transformación y escalado |

Adicionalmente, K-Medoids es compatible con cualquier métrica de distancia (no solo euclidiana), lo que otorga flexibilidad para iteraciones futuras del modelo.

### 3.2 Configuración del Backend y Parámetros

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| **Algoritmo** | PAM (Partitioning Around Medoids) | Método clásico de Kaufman y Rousseeuw. Minimiza la suma de distancias intra-cluster mediante intercambios iterativos de medoides. Es el estándar de referencia para K-Medoids |
| **Backend** | `pyclustering` (con fallback a `sklearn_extra`) | `pyclustering` ofrece implementación nativa de PAM con control granular sobre la inicialización de medoides. `sklearn_extra` es el fallback si `pyclustering` no está disponible en el entorno |
| **Métrica de distancia** | Euclidiana | Adecuada para features continuas en espacio transformado (log1p + RobustScaler). La distancia euclidiana en el espacio RFM transformado captura diferencias proporcionales entre comercios |
| **Método de inicialización** | Selección aleatoria con semilla fija (`random_state=42`) | La inicialización aleatoria con semilla fija garantiza **reproducibilidad exacta** entre ejecuciones. La estabilidad de la solución se verifica posteriormente mediante el test ARI con 20 corridas independientes |
| **Rango de K evaluado** | K = 2, 3, 4, 5, 6, 7, 8 | Rango suficiente para capturar desde una solución binaria trivial hasta una segmentación con granularidad excesiva. Más de 8 clusters con 3,015 comercios resultaría en segmentos demasiado pequeños para campañas diferenciadas |

### 3.3 Evaluación del Rango de Clusters (K = 2 a 8)

Para cada valor de K en el rango [2, 8] se entrenó un modelo K-Medoids y se calcularon cuatro métricas de evaluación. Como referencia comparativa, se entrenó también K-Means con los mismos parámetros de inicialización (`n_init=50`, `random_state=42`).

#### Métricas Utilizadas

| Métrica | Definición | Interpretación | Dirección deseada |
|---------|-----------|----------------|-------------------|
| **Costo PAM** | Suma de distancias euclidianas de cada punto a su medoide | Mide la compacidad intra-cluster. Decrece monótonamente con K | ↓ Menor = mejor |
| **Silhouette Score** | (Cohesión intra − Separación inter) / max(cohesión, separación). Rango: [−1, 1] | Evalúa qué tan bien asignado está cada punto a su cluster. Valores cercanos a 1 indican clusters bien separados y cohesivos | ↑ Mayor = mejor |
| **Davies-Bouldin Index** | Promedio de la máxima similitud entre cada cluster y su más cercano. La similitud es el ratio de dispersiones internas sobre distancia entre centroides | Mide la separación promedio entre clusters. Valores cercanos a 0 indican clusters compactos y bien separados | ↓ Menor = mejor |
| **ARI (Adjusted Rand Index)** | Medida de similitud entre dos asignaciones de clustering, ajustada por azar. Rango: [−1, 1], donde 0 = asignación aleatoria | Evalúa la estabilidad de la solución ante diferentes inicializaciones. Se calcula sobre 20 corridas independientes con distintas semillas | ↑ Mayor = más estable |

#### Resultados de la Evaluación

Para K=2, el Silhouette Score alcanza su valor máximo —pero esta es una **solución trivial**: divide la cartera en dos macro-clusters (comercios activos vs. inactivos, o alto valor vs. bajo valor) sin granularidad suficiente para diseñar campañas de marketing diferenciadas por segmento. Una segmentación binaria no permite priorizar esfuerzos comerciales ni asignar presupuestos de manera diferenciada.

Para el resto de valores de K, se observa el comportamiento esperado:

- El **Costo PAM** decrece monótonamente con K (más clusters → puntos más cerca de su medoide).
- El **Silhouette Score** presenta un máximo local o meseta en K=4 antes de degradarse para K mayores.
- El **Davies-Bouldin Index** alcanza un mínimo relativo en K=4, indicando la mejor relación compacidad/separación.

### 3.4 Justificación de K=4 como Solución Óptima

La selección de **K=4** como número óptimo de clusters se fundamenta en criterios tanto estadísticos como de negocio:

**Criterios estadísticos:**

- K=4 ofrece el mejor equilibrio en la curva de Silhouette: captura la estructura subyacente de los datos sin sobre-ajustar (K=5+ produce segmentos excesivamente fragmentados).
- El Davies-Bouldin Index alcanza su valor más bajo en K=4, indicando máxima separación entre clusters y máxima compacidad interna.
- El test de estabilidad ARI con 20 corridas independientes produce un **ARI medio de 0.9061** (σ = 0.176, rango [0.358, 1.000]), clasificando la solución como de **alta estabilidad** — el resultado es robusto y reproducible.
- **Métricas finales para K=4:** Silhouette Score = **0.3994**, Davies-Bouldin Index = **0.8947**, Costo PAM = **1,870.30**.

**Criterios de negocio:**

Cuatro segmentos permiten diseñar estrategias comerciales **diferenciadas y accionables**:

1. **Champions / VIP:** defensa y fidelización del segmento que concentra el 83.35% de las comisiones.
2. **Core / Crecimiento:** expansión y cross-selling sobre comercios con potencial de crecimiento comprobado.
3. **En Riesgo / Churn:** intervención temprana para prevenir la pérdida de comercios inactivos.
4. **Ocasional / Ticket Alto:** activación de comercios con transacciones esporádicas pero de alto valor unitario.

K=3 no ofrecería suficiente granularidad (fusionaría segmentos con comportamientos distintos, perdiendo la capacidad de diferenciar estrategias). K=5+ produciría segmentos con tamaños menores a 200 comercios, insuficientes para justificar campañas independientes con retorno sobre inversión positivo.

### 3.5 Importancia e Interpretación de los Medoides Reales

Una ventaja distintiva de K-Medoids frente a K-Means es que el representante de cada cluster es un **comercio real** —identificable por su `id_comercio`— y no un centroide matemático abstracto.

Los medoides identificados por el modelo son:

| Cluster | Segmento | ID Medoide |
|---------|----------|------------|
| 0 | Core / Crecimiento | I06YDZ157S |
| 1 | En Riesgo / Churn | 0773NFXBJD |
| 2 | Champions / VIP | 96FVKJJ572 |
| 3 | Ocasional / Ticket Alto | JR7DNNZ0D0 |

**Implicaciones prácticas:**

- **Auditoría:** el equipo de ciencia de datos puede extraer el comercio medoide, examinar su historial transaccional completo y verificar que efectivamente tipifica el comportamiento del segmento.
- **Comunicación con negocio:** en lugar de describir un segmento como "comercios con coordenadas promedio (μ_recencia, μ_frecuencia, μ_monetary)", se puede decir: "El segmento Champions está representado por el comercio ID 96FVKJJ572, cuyo comportamiento transaccional define el perfil ideal del segmento."
- **Trazabilidad:** los IDs de los medoides se registran en el archivo de metadata JSON, permitiendo reproducir exactamente la misma asignación de clusters en cualquier momento futuro.

### 3.6 Registro de Artefactos Exportados

El notebook de segmentación produce y exporta el siguiente conjunto de artefactos:

| Artefacto | Ruta | Contenido |
|-----------|------|-----------|
| Dataset final | `data/processed/comercios_final_k_medoids.csv` | 3,015 comercios con columna `cluster` y `segmento` asignados |
| Perfiles de segmentos | `data/processed/perfiles_segmentos_k_medoids.csv` | Tabla resumen con medias, medianas y percentiles (P10, P25, P75, P90) de cada variable RFM por cluster, más comisión total y porcentaje |
| Modelo entrenado | `data/processed/model_k_medoids.pkl` | Objeto K-Medoids serializado con `joblib` (clusters, medoides, labels) |
| Scaler ajustado | `data/processed/scaler_k_medoids.pkl` | Objeto `RobustScaler` ajustado sobre el feature space transformado |
| Metadata JSON | `data/processed/metadata_segmentation_k_medoids.json` | Documenta timestamp, backend, features, parámetros, métricas finales (Silhouette=0.3994, DB=0.8947, Costo=1870.30, ARI=0.9061), reglas de asignación de nombres de segmentos, percentiles globales usados como umbrales, IDs de los medoides |
| Gráficos (9 archivos) | `data/processed/plots/kmedoids_*.png` | Distribuciones raw, matriz de correlación, curvas Elbow/Silhouette/DB, gráfico de silueta por muestra, proyección PCA 2D, visualización 3D RFM, perfiles de segmentos |
| Simulación financiera | `data/processed/simulacion_puente_powerbi.csv` | Tabla waterfall con desglose por escenario × componente, incluyendo KPIs operativos y contribución porcentual al cierre del gap |

---

## 4. Perfilamiento de Segmentos

### 4.1 Metodología de Asignación de Nombres

Los nombres de los segmentos se asignan mediante un **sistema automatizado basado en reglas** (`asignar_nombre_segmento()`), que opera sobre las **medias agregadas de cada cluster** —no sobre comercios individuales— y utiliza **percentiles globales** calculados una única vez sobre el dataset completo. Este enfoque garantiza:

- **Reproducibilidad:** la misma lógica produce los mismos nombres en cualquier ejecución, eliminando el sesgo de la interpretación manual del analista.
- **Adaptabilidad:** los umbrales (percentiles) se ajustan automáticamente a la distribución real de los datos, sin valores hardcodeados.
- **Auditabilidad:** cada asignación queda registrada con la regla específica que la activó.

**Percentiles globales de referencia:**

| Variable | P25 | P50 | P75 |
|----------|-----|-----|-----|
| `recency_days` | 0.0 | 0.0 | 30.0 |
| `frequency` | 3.0 | 10.0 | 23.0 |
| `monetary` | COP $325,700 | COP $1,554,000 | COP $4,976,200 |
| `ticket_avg` | COP $39,802 | COP $120,727 | COP $301,231 |

**Reglas de asignación aplicadas (en orden de prioridad):**

| Prioridad | Regla | Condición | Nombre asignado |
|-----------|-------|-----------|-----------------|
| 1 | Champions / VIP | `frequency_mean > P75` **Y** `recency_days_mean < umbral_recencia`¹ **Y** `monetary_mean > P75` | Champions / VIP |
| 2 | En Riesgo / Churn | `recency_days_mean > P75` **Y** `frequency_mean < P25` | En Riesgo / Churn |
| 3 | Ocasional / Ticket Alto | `frequency_mean < P50`² **Y** `ticket_avg_mean > P75` | Ocasional / Ticket Alto |
| 4 | Core / Crecimiento | Catch-all — no cumple ninguna regla anterior | Core / Crecimiento |

> ¹ **Ajuste por floor effect**: cuando P25(recency) = 0 (más del 50% de los comercios transaccionaron el mismo día del corte), el umbral se fija en 1.0 día como proxy — ningún cluster puede tener recencia media negativa.
>
> ² **Ajuste de frecuencia para Ocasional**: P25(frequency) = 3 tx/6m es demasiado restrictivo. Comercios con 4-5 transacciones en 6 meses también son "ocasionales". Se eleva el umbral a P50 (mediana de frecuencia): transaccionan menos que la mitad de la cartera.

### 4.2 Tabla Resumen de Métricas por Segmento

| Métrica | Champions / VIP | Core / Crecimiento | Ocasional / Ticket Alto | En Riesgo / Churn |
|---------|:---:|:---:|:---:|:---:|
| **Comercios (N)** | 1,076 | 1,169 | 363 | 407 |
| **% Comercios** | 35.7% | 38.8% | 12.0% | 13.5% |
| **Recencia media (días)** | 0.7 | 3.4 | 55.5 | 77.3 |
| **Recencia mediana (días)** | 0.0 | 0.0 | 61.0 | 61.0 |
| **Frecuencia media (tx)** | 51.7 | 8.9 | 4.4 | 1.3 |
| **Frecuencia mediana (tx)** | 32.0 | 8.0 | 3.0 | 1.0 |
| **Monetary medio (COP)** | $10,249,381 | $1,563,181 | $1,026,470 | $5,770 |
| **Monetary mediano (COP)** | $6,684,200 | $1,133,000 | $320,720 | $2,000 |
| **Ticket promedio (COP)** | $322,247 | $237,969 | $332,203 | $3,086 |
| **Comisión total (COP)** | $551,416,717 | $91,367,952 | $18,630,437 | $117,410 |
| **% Comisión total** | **83.35%** | 13.81% | 2.82% | 0.02% |
| **Terminales promedio** | 1.04 | 1.01 | 1.01 | 1.01 |
| **Regla de asignación** | R1: Champions/VIP | R4: Core/Crecimiento (catch-all) | R3: Ocasional/Ticket Alto | R2: En Riesgo/Churn |

### 4.3 Descripción Detallada de Cada Segmento

#### Segmento 2 — Champions / VIP (1,076 comercios, 83.35% de las comisiones)

**Perfil:** Recencia prácticamente nula (0.7 días en promedio, con mediana y percentil 75 en 0 días — la mayoría transaccionó el mismo día del corte). Frecuencia muy alta (51.7 transacciones en 6 meses, mediana de 32). Monetary muy elevado ($10.2 M en promedio, con el percentil 90 superando los $21.6 M).

**Interpretación analítica:** Este es el segmento que sostiene el negocio. Con solo el 35.7% de los comercios, genera el 83.35% de las comisiones totales (COP $551.4 M de COP $661.5 M). La distribución de `monetary` dentro del segmento es amplia (P10 = $2.4 M, P90 = $21.6 M), indicando que contiene tanto comercios grandes como medianos con alta actividad. La práctica ausencia de recencia (mediana = 0 días) sugiere una operación diaria o casi diaria. El ticket promedio ($322,247) y la frecuencia alta (51.7 tx) indican comercios que procesan volúmenes significativos de forma consistente.

**Implicación operativa:** Este segmento **no debe ser perturbado** con tácticas agresivas de upselling que puedan generar fricción. La estrategia prioritaria es **defensa y blindaje**: cualquier erosión de este segmento tiene un impacto desproporcionado sobre el P&L (una pérdida del 5% en VIP equivale a COP $27.6 M, más que toda la comisión del segmento Ocasional).

#### Segmento 0 — Core / Crecimiento (1,169 comercios, 13.81% de las comisiones)

**Perfil:** Recencia muy baja (3.4 días en promedio, mediana 0 días). Frecuencia media-alta (8.9 transacciones, mediana 8). Monetary medio ($1.56 M, mediana $1.13 M). Ticket promedio de COP $237,969.

**Interpretación analítica:** Comercios activos y regulares, con volumen transaccional moderado pero consistente. La recencia cercana a cero (mediana = 0) confirma que son comercios que utilizan la plataforma de forma continua. Su frecuencia (8.9 tx en 6 meses) sugiere aproximadamente 1.5 transacciones por mes — espacio claro de expansión. Con 1,169 comercios, es el segmento más numeroso (38.8% de la cartera). La media de monetary ($1.56 M) y la mediana ($1.13 M) indican que la mayoría tiene un tamaño de negocio moderado, con algunos comercios más grandes elevando el promedio (P90 = $3.38 M).

**Implicación operativa:** Segmento con mayor potencial de crecimiento marginal. Pequeñas mejoras en frecuencia (de 1.5 tx/mes a 3 tx/mes) o en ticket promedio tienen un efecto multiplicador por la gran base de comercios. Es el candidato ideal para estrategias de cross-selling (segunda terminal) y upselling (incentivos por volumen). La asignación vía catch-all (R4) indica que estos comercios tienen métricas intermedias en todas las dimensiones — no son extremos en recencia, frecuencia ni monetary, lo que los hace maleables a intervenciones de crecimiento.

#### Segmento 3 — Ocasional / Ticket Alto (363 comercios, 2.82% de las comisiones)

**Perfil:** Recencia moderada-alta (55.5 días en promedio). Frecuencia baja (4.4 transacciones en 6 meses, mediana 3). Monetary medio de $1.03 M. **Ticket promedio más alto de todos los segmentos:** COP $332,203 (vs. $322,247 de VIP y $237,969 de Core). La mediana del ticket es $101,333.

**Interpretación analítica:** Comercios que transaccionan esporádicamente pero con valores unitarios elevados. La dispersión del ticket es considerable (P25 = $40,198, P90 = $606,800), sugiriendo una mezcla de negocios con comportamientos distintos dentro del segmento. La recencia elevada (55.5 días) indica que en promedio pasan casi 2 meses entre transacciones, pero cuando transaccionan, los montos son altos. La mediana de recencia (61 días) revela que la mayoría está cerca de los 2 meses de inactividad, con el P25 en 30 días — algunos transaccionan mensualmente pero la mayoría está en ciclos más largos.

**Implicación operativa:** La palanca de crecimiento aquí es la **frecuencia**, no el ticket. El objetivo es convertir transacciones esporádicas en recurrentes. Estrategias de facturación recurrente, suscripciones y recordatorios automatizados pueden reducir la recencia sin afectar el ticket promedio. La integración con soluciones de cobro recurrente es particularmente relevante para este perfil de negocio.

#### Segmento 1 — En Riesgo / Churn (407 comercios, 0.02% de las comisiones)

**Perfil:** Recencia muy alta (77.3 días en promedio, P25 = 61 días, P90 = 120 días). Frecuencia casi nula (1.3 transacciones en 6 meses, mediana 1). Monetary mínimo ($5,770 en promedio, mediana $2,000). Ticket promedio de COP $3,086.

**Interpretación analítica:** Comercios prácticamente inactivos. Con una recencia superior a 2.5 meses (77 días) y apenas 1 transacción en todo el período, estos comercios están al borde del abandono definitivo. Su aporte a las comisiones es insignificante (0.02%, COP $117,410 entre 407 comercios). Sin embargo, representan el 13.5% de la cartera — son comercios que en algún momento estuvieron activos y cuyo costo de adquisición ya fue incurrido. El P90 de recencia (120 días) indica que algunos llevan 4 meses sin transaccionar — posiblemente ya desertaron.

**Implicación operativa:** Aunque su contribución actual es mínima, recuperar aunque sea el 10% de estos comercios (41 comercios) a niveles de actividad similares a Core representaría COP $3.2 M adicionales con costo de adquisición cero. La intervención debe ser temprana (D+45) para maximizar la probabilidad de reactivación. Pasado D+90, la probabilidad de recuperación decae drásticamente. El verdadero valor de este segmento no es monetario directo sino **preventivo**: las tácticas de retención evitan que comercios actualmente en Core o VIP migren a este estado.

---

## 5. Conexión con el Objetivo de Negocio

### 5.1 Segmentos con Mayor Aporte a las Comisiones

El análisis de concentración revela una **distribución de Pareto extrema:**

- **Champions / VIP** (35.7% de comercios) → **83.35%** de las comisiones (COP $551.4 M)
- **Champions / VIP + Core / Crecimiento** (74.5% de comercios) → **97.16%** de las comisiones (COP $642.8 M)
- **En Riesgo / Churn + Ocasional / Ticket Alto** (25.5% de comercios) → **2.84%** de las comisiones (COP $18.7 M)

La implicación es contundente: **cualquier estrategia de crecimiento en comisiones debe priorizar la defensa de VIP y la expansión de Core.** El 97.16% de las comisiones está concentrado en 2,245 comercios.

### 5.2 Tácticas Específicas por Segmento para Cerrar la Brecha de +COP $330.8 M

**Contexto financiero:**

| Indicador | Valor |
|-----------|-------|
| Comisión total actual (corte May 2021) | COP $661,532,517 |
| Meta +50% (Enero 2022) | COP $992,298,775 |
| Gap absoluto a cerrar | COP $330,766,258 |
| Ventana de ejecución | Junio – Diciembre 2021 (7 meses) |

**Palancas por segmento (Escenario Intermedio):**

#### Champions / VIP — Blindaje y Fidelización (+15% uplift)

- **Táctica:** Programa de fidelización premium con tasa de comisión preferencial por permanencia.
- **Mecanismo:** Asignación de Account Manager dedicado al top 50 de VIP, reportes mensuales personalizados, acceso anticipado a nuevas funcionalidades de API.
- **Impacto proyectado:** +COP $82,712,508 adicionales (25.01% del gap).
- **Riesgo mitigado:** Sin intervención, la pérdida de tan solo 5 comercios VIP (0.5% del segmento) erosiona ~COP $2.7 M — más que toda la comisión del segmento Ocasional.

#### Core / Crecimiento — Expansión y Cross-Selling (+20% uplift)

- **Táctica:** Cross-selling de segunda terminal con incentivo por adopción + planes de comisión escalonada por volumen transaccional.
- **Mecanismo:** Bonificación del 50% en la comisión incremental durante los primeros 3 meses para comercios que dupliquen su frecuencia.
- **Impacto proyectado:** +COP $18,273,590 adicionales (5.52% del gap).
- **Potencial latente:** 1,169 comercios con frecuencia media de 8.9 tx/6m y recencia de 3.4 días. Cada comercio que duplica su actividad transaccional duplica su aporte en comisiones sin costo incremental de adquisición.

#### En Riesgo / Churn — Retención (10% de retención)

- **Táctica:** Alerta temprana de churn D+45 + descuento temporal del 30% en comisión por 60 días de reactivación + llamada de diagnóstico.
- **Impacto proyectado:** +COP $11,741 (0.004% del gap). La contribución de este segmento no es monetaria directa sino **preventiva**: evitar que comercios "Core" migren a "Riesgo".

#### Ocasional / Ticket Alto — Reactivación (+5% uplift)

- **Táctica:** Incentivos por frecuencia (>8 tx/mes) + integración con soluciones de cobro recurrente.
- **Impacto proyectado:** +COP $931,522 adicionales (0.28% del gap).

**Cierre del gap con cartera actual:** Las palancas sobre la cartera actual generan COP $101,929,361 en el Escenario Intermedio, cerrando el **30.8% del gap**. Los COP $228,836,897 restantes (69.2%) requieren estrategias complementarias ejecutadas en paralelo durante el segundo semestre de 2021:

- **Captación neta de comercios:** ~223 comercios Look-Alike VIP (~32/mes) + ~1,464 comercios Look-Alike Core (~209/mes).
- **Migración Core → VIP:** comercios que superen 30 tx/mes migran a condiciones preferenciales.
- **Revisión de pricing:** ajuste de estructura de comisiones por volumen en segmentos de menor contribución.

### 5.3 Alineación Directa con la Meta del 50%

La segmentación K-Medoids proporciona el **marco de priorización** indispensable para alcanzar la meta en una ventana de solo 7 meses. Sin segmentación, el equipo comercial trataría a los 3,015 comercios como una masa homogénea, diluyendo el esfuerzo. Con segmentación:

1. **Defensa del 83.35%** de las comisiones concentradas en Champions/VIP (Fase 1: Jun–Ago 2021).
2. **Expansión del 13.81%** en Core/Crecimiento con tácticas de cross-selling (Fase 2: Sep–Nov 2021).
3. **Captación neta** de ~1,687 comercios con perfil Look-Alike de VIP y Core como palanca complementaria (Fase 1–2: Jun–Dic 2021, en paralelo).
4. **KPIs de seguimiento mensual:** comisión total acumulada vs. meta proporcional del mes, % de comercios que ascienden de segmento (Core → VIP), `recency_days` promedio del segmento En Riesgo (objetivo: reducir de 77d a <45d), Net Revenue Retention (NRR) por cohorte de segmento, tasa de adopción de segunda terminal en Core.

---

## 6. Respuesta a la pregunta: Describa el modelo de segmentación implementado y los resultados obtenidos

### Modelo Seleccionado

Se implementó un modelo de segmentación no supervisada basado en **K-Medoids (PAM — Partitioning Around Medoids)** sobre el espacio de features RFM clásico aplicado a 3,015 comercios activos en los últimos 6 meses (ventana de análisis con corte en mayo de 2021).

### Justificación Técnica

K-Medoids fue seleccionado sobre K-Means por tres razones principales: (1) **resistencia a outliers** — el dataset contiene comercios con `monetary` de hasta COP $217 M que en K-Means distorsionarían los centroides, mientras que en K-Medoids el representante de cada cluster es un punto real de la muestra, inmune a valores extremos; (2) **interpretabilidad comercial** — cada cluster está representado por un `id_comercio` real y trazable (medoides: 96FVKJJ572, I06YDZ157S, JR7DNNZ0D0, 0773NFXBJD), permitiendo al equipo de negocio examinar el comercio representativo y entender qué tipifica al segmento; (3) **auditabilidad** — a diferencia de un centroide matemático abstracto, el medoide es un comercio concreto cuyo historial transaccional puede ser verificado.

### Variables Utilizadas

Se utilizó el **Feature Set A — RFM Clásico** compuesto por tres dimensiones:

- **Recencia** (`recency_days`): días desde la última transacción del comercio hasta la fecha de corte del análisis. No se aplicó transformación logarítmica para preservar su interpretación lineal en el contexto de riesgo de churn.
- **Frecuencia** (`frequency`): total de transacciones del comercio en el período de 6 meses. Se aplicó transformación `log1p` para corregir la asimetría severa (>3.0) típica de distribuciones financieras con cola larga.
- **Valor Monetario** (`monetary`): monto acumulado de ventas (suma de todos los terminales del comercio). Se aplicó transformación `log1p` por la misma razón que frecuencia — la distribución log-normal de los montos transaccionales.

### Preprocesamiento Aplicado

- **Transformación logarítmica** (`np.log1p`): aplicada exclusivamente a `frequency` y `monetary` para normalizar sus distribuciones con asimetría > 3.0.
- **Escalado robusto** (`RobustScaler`): utiliza mediana (Q2) y rango intercuartílico (IQR) en lugar de media y desviación estándar. Esta decisión fue crítica, ya que `StandardScaler` habría comprimido el 99% de los comercios cerca de cero debido al efecto de los outliers extremos en `monetary` (COP $217 M) y `ticket_avg` (COP $29 M).
- **Preservación de outliers**: los comercios con valores extremos se mantuvieron en la muestra. La decisión se fundamenta en que (a) K-Medoids es resistente a outliers por diseño, (b) RobustScaler no se ve afectado por ellos, y (c) estos valores extremos representan comportamiento real del negocio — los comercios más grandes son precisamente los que generan más comisiones.

### K Óptimo Seleccionado

Se evaluó el rango K = 2 a 8 mediante cuatro métricas: Costo PAM, Silhouette Score, Davies-Bouldin Index y estabilidad ARI (20 corridas independientes). **K=2 fue descartado** pese a tener el Silhouette Score más alto, por ser una solución trivial no accionable (divide la cartera en solo dos macro-clusters, sin granularidad para campañas diferenciadas). **K=4 fue seleccionado** como el punto de equilibrio óptimo entre calidad estadística y utilidad comercial, con validación de alta estabilidad (ARI medio = 0.9061, σ = 0.176). Las métricas finales del modelo son: **Silhouette Score = 0.3994**, **Davies-Bouldin Index = 0.8947**, **Costo PAM = 1,870.30**.

### Segmentos Resultantes y Hallazgos Clave

El modelo identificó cuatro segmentos con comportamientos transaccionales claramente diferenciados:

1. **Champions / VIP** (1,076 comercios, 35.7%): recencia de 0.7 días, frecuencia de 51.7 tx, monetary de COP $10.2 M. **Concentran el 83.35% de las comisiones totales (COP $551.4 M).** Son el segmento que sostiene el negocio y cuya defensa es la máxima prioridad.

2. **Core / Crecimiento** (1,169 comercios, 38.8%): recencia de 3.4 días, frecuencia de 8.9 tx, monetary de COP $1.56 M. Aportan el 13.81% de las comisiones. Representan la base de comercios activos con mayor potencial de expansión vía cross-selling.

3. **Ocasional / Ticket Alto** (363 comercios, 12.0%): recencia de 55.5 días, frecuencia de 4.4 tx, ticket promedio de COP $332,203 — el más alto de todos los segmentos. Aportan el 2.82% de las comisiones. La palanca de crecimiento es aumentar la frecuencia transaccional.

4. **En Riesgo / Churn** (407 comercios, 13.5%): recencia de 77.3 días, frecuencia de 1.3 tx, monetary de COP $5,770. Aportan solo el 0.02% de las comisiones. Son comercios al borde del abandono definitivo que requieren intervención temprana de retención.

**Hallazgo de concentración (Principio de Pareto):** Los dos segmentos superiores (VIP + Core) representan el 74.5% de los comercios pero concentran el **97.16%** de las comisiones. Esto valida que la estrategia de crecimiento debe priorizar defender VIP y expandir Core antes que intentar recuperar comercios inactivos.

### Plan de Acción Comercial

La meta de negocio —incrementar las comisiones en un 50% para enero de 2022, cerrando una brecha de COP $330.8 M en 7 meses— se aborda mediante un plan en tres fases:

- **Fase 1 — Blindaje (Jun–Ago 2021):** Defender el 83.35% de comisiones en VIP mediante programa de fidelización y Account Manager dedicado. Activar alertas de churn en Riesgo (D+45). Impacto: +COP $82.7 M (25.0% del gap).

- **Fase 2 — Expansión (Sep–Nov 2021):** Cross-selling de segunda terminal en Core, incentivos por frecuencia en Ocasional, migración de Core a VIP para comercios que superen 30 tx/mes. Impacto: +COP $19.2 M (5.8% del gap).

- **En paralelo (Jun–Dic 2021):** Captación neta de ~1,687 comercios Look-Alike y revisión de pricing en segmentos de menor contribución. Impacto: +COP $228.8 M (69.2% del gap).

**KPIs de seguimiento:** comisión total acumulada vs. meta proporcional mensual, tasa de ascenso de segmento (Core → VIP), evolución de `recency_days` en Riesgo (objetivo < 45 días), Net Revenue Retention (NRR) por cohorte, tasa de adopción de segunda terminal en Core.

---

*Documento generado a partir del análisis del código fuente (`src/etl.py`, `src/segmentation.py`, `src/simulacion_financiera.py`, `src/gen_kmedoids_nb.py`), notebook de segmentación (`notebooks/segmentation_evaluation_k_medoids.ipynb`) y datasets procesados (`comercios_segmented_prep.csv`, `comercios_final_k_medoids.csv`, `perfiles_segmentos_k_medoids.csv`, `metadata_segmentation_k_medoids.json`, `simulacion_puente_powerbi.csv`).*
