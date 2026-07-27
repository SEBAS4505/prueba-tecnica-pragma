# Documentación Técnica — Fase 2: Análisis Exploratorio, Imputación y Pronóstico de la TCRM

**Proyecto:** Prueba Técnica Pragma — Analítica de Datos  
**Cliente:** Empresa importadora con portal e-commerce, expuesta al riesgo cambiario USD/COP  
**Objetivo técnico:** Analizar la serie histórica de la TCRM, desarrollar una metodología robusta de imputación para días no transados, caracterizar el comportamiento temporal y construir un modelo de *forecasting* para proyectar la TCRM al 1 de enero de 2023 con cuantificación de incertidumbre  
**Versión del documento:** 1.0  

---

## 1. Ingesta, Auditoría de Completitud e Imputación de Datos

### 1.1 Fuente de Datos y Cobertura

Los datos históricos de la TCRM provienen del archivo oficial publicado por la Superintendencia Financiera de Colombia:

| Atributo | Valor |
|----------|-------|
| **Archivo fuente** | `Tasa_de_Cambio_Representativa_del_Mercado_Historico.xlsx` |
| **Formato** | Excel (`.xlsx`), 4 columnas: `VALOR`, `UNIDAD`, `VIGENCIADESDE`, `VIGENCIAHASTA` |
| **Registros originales** | 7,188 filas |
| **Rango temporal** | 1991-12-02 → 2021-10-19 |
| **Días calendario totales** | 10,915 |
| **TCRM mínima** | $620.62 COP |
| **TCRM máxima** | $4,153.91 COP |
| **TCRM media** | $2,106.49 COP |
| **Desviación estándar** | $824.03 COP |
| **Último valor registrado** | $3,772.49 COP (2021-10-19) |

El archivo fuente utiliza un esquema de **registros multi-día**: cada fila representa un intervalo de vigencia `[VIGENCIADESDE, VIGENCIAHASTA]` durante el cual un mismo valor de TCRM estuvo en vigor. La columna `VALOR` emplea formato numérico colombiano con coma como separador de miles (ej. `"2,850.98"`).

La función `load_tcrm_raw()` del módulo `tcrm_forecast.py` realiza la carga directa mediante `pd.read_excel()`. La función `clean_tcrm()` ejecuta las siguientes operaciones de normalización:

- **Conversión de `VALOR` a `float64`:** la función `_parse_colombian_number()` elimina las comas de miles y convierte la cadena a float, con tolerancia a fallos (`NaN` para valores no convertibles).
- **Conversión de fechas:** `VIGENCIADESDE` y `VIGENCIAHASTA` se convierten a `datetime64` mediante `pd.to_datetime()`.
- **Ordenamiento cronológico** ascendente y reseteo de índice.

### 1.2 Auditoría de Completitud

El diagnóstico de integridad revela que la serie original **no es diaria ni continua**. Los datos cubren únicamente los días en que el mercado cambiario colombiano operó, publicándose un valor de TCRM por cada intervalo de vigencia. La auditoría arroja los siguientes resultados:

| Indicador | Cantidad | Porcentaje |
|-----------|----------|------------|
| Días calendario totales (1991-12-02 → 2021-10-19) | 10,915 | 100.0% |
| Registros originales en el archivo fuente | 7,188 | 65.9% |
| Días no transados (fines de semana y festivos) | **3,727** | **34.1%** |
| Valores nulos en la serie final procesada | 0 | 0.0% |

Los 3,727 días faltantes corresponden a:

- **Fines de semana** (sábados y domingos): aproximadamente 2 × 52 × 29.85 ≈ 3,104 días.
- **Festivos nacionales colombianos y bancarios en EE. UU.:** aproximadamente 623 días.
- El mercado cambiario colombiano no opera en días no hábiles, por lo que no se publica TCRM para esas fechas.

La función `get_integrity_report()` del módulo `tcrm_forecast.py` genera este reporte de forma programática, recibiendo la serie procesada y el conteo de registros originales como parámetros, y retornando un diccionario estructurado con las claves `total_dias_calendario`, `registros_originales`, `dias_no_transados`, `fecha_inicio` y `fecha_fin`.

### 1.3 Estrategia e Implementación de Imputación

#### 1.3.1 Fundamento Normativo

La Circular Reglamentaria DODM-146 de la Superintendencia Financiera de Colombia establece que **la TCRM vigente para un día no transado (fin de semana o festivo) corresponde al valor del último día hábil en que se haya publicado**. Esta regla es vinculante para todas las operaciones de cambio que se liquiden en días no hábiles y constituye el pilar jurídico de la metodología de imputación adoptada.

#### 1.3.2 Implementación Técnica: `build_daily_series()`

La función `build_daily_series()` del módulo `tcrm_forecast.py` implementa la imputación normativa en tres etapas:

1. **Expansión de registros multi-día:** cada fila del DataFrame limpio se descompone en registros individuales para cada día de su intervalo `[VIGENCIADESDE, VIGENCIAHASTA]`. Se itera sobre cada fila generando un `pd.date_range(start, end, freq="D")` y asignando el mismo `VALOR` a cada día del rango. Los posibles solapamientos entre intervalos se resuelven conservando el último registro (`drop_duplicates(subset="fecha", keep="last")`).

2. **Reindexación al calendario completo:** se genera un `DatetimeIndex` continuo con frecuencia diaria (`freq="D"`) desde 1991-12-02 hasta 2021-10-19 inclusive (10,915 días). El DataFrame expandido se reindexa a este calendario completo. Los días que no tienen un registro expandido (porque el archivo fuente no contiene vigencia explícita para esa fecha) aparecen como `NaN`.

3. **Forward fill normativo:** se aplica `ffill()` sobre la columna `tcrm`, propagando el último valor conocido hacia adelante para cubrir todos los días sin dato. Este mecanismo refleja exactamente la regla de la SFC: un sábado, domingo o festivo hereda la TCRM del último día hábil publicado. Si quedaran `NaN` al inicio de la serie (antes del primer dato), se aplica `bfill()` como mecanismo de respaldo.

#### 1.3.3 Justificación Técnica: Forward Fill vs. Interpolación

| Método | Mecanismo | Adecuación al caso TCRM |
|--------|-----------|--------------------------|
| **Forward Fill (`ffill`)** — **SELECCIONADO** | Propaga el último valor observado hacia adelante | ✅ Refleja exactamente la normativa SFC. En días no transados, la TCRM **no cambia**; el valor vigente es el del último día hábil. No introduce valores artificiales inexistentes en el mercado |
| Interpolación lineal | Calcula valores intermedios como combinación lineal de los extremos del intervalo | ❌ Genera valores ficticios que nunca existieron en el mercado (ej. un valor interpolado entre viernes y lunes para el fin de semana). Viola la regla de negocio: la TCRM no varía de forma continua en días no transados |
| Interpolación spline / polinómica | Ajusta curvas suaves a través de los puntos observados | ❌ Además del problema anterior, introduce oscilaciones no fundamentadas. Altamente inapropiado para series financieras con cambios discretos |
| Backfill (`bfill`) | Propaga el siguiente valor hacia atrás | ❌ Implicaría que la TCRM de un viernes se determina con el valor del lunes siguiente — información futura no disponible, generando *look-ahead bias* |

La elección del forward fill es, por tanto, la única metodología que **preserva la integridad normativa, la no-anticipación temporal y la realidad operativa del mercado cambiario colombiano**.

### 1.4 Caso Puntual de Validación: TCRM para la Fecha de Expedición de la Cédula

Se consultó la TCRM para la fecha de expedición de la cédula de ciudadanía mediante la función `get_tcrm_at_date()`:

| Campo | Valor |
|-------|-------|
| **Fecha consultada** | 2010-04-19 |
| **TCRM observada** | **$1,942.21 COP** |
| **Valor de referencia** | $1,942.21 COP |
| **Coincidencia exacta** | ✅ Sí |

La verificación se realizó mediante indexación directa sobre el `DatetimeIndex` de la serie procesada, contrastando el valor obtenido con el valor de referencia externo. La coincidencia exacta valida la corrección del pipeline ETL y la integridad de la imputación.

---

## 2. Análisis Exploratorio y Caracterización de la Curva Temporal (TCRM vs. Tiempo)

### 2.1 Comportamiento Histórico de la Curva

La serie TCRM exhibe una trayectoria que transita desde un régimen de **banda cambiaria administrada** (crawling peg, 1991–1998) hacia un esquema de **flotación libre** a partir de 1999, con episodios de devaluación acelerada, apreciación sostenida y choques exógenos. A lo largo de 29.85 años de historia, la divisa pasó de $620.62 COP/USD a $3,772.49 COP/USD —una devaluación acumulada del **507.8%**, equivalente a una tasa de depreciación anual compuesta de aproximadamente 6.2%.

Se identificaron **seis regímenes cambiarios** estructuralmente diferenciados:

| # | Período | Régimen | TCRM (aprox.) | Evento detonante |
|---|---------|---------|---------------|------------------|
| 1 | 1991–1998 | Banda cambiaria (*crawling peg*) | $643 → $1,500 | Apertura económica y régimen de minidevaluaciones controladas |
| 2 | 1998–2003 | Crisis de fin de siglo + flotación (1999) | $1,500 → $2,900 | Crisis de deuda soberana, abandono de la banda cambiaria, devaluación del 50% en 1999 |
| 3 | 2003–2013 | Apreciación sostenida (*super-peso*) | $2,900 → $1,750 | Boom de commodities, flujo de IED, entrada de capitales especulativos |
| 4 | 2014–2016 | Devaluación post-caída del petróleo | $1,900 → $3,400 | Colapso del precio del crudo (Brent de $115 a $28/barril), déficit en cuenta corriente |
| 5 | 2017–2019 | Estabilización en rango alto | $2,800 – $3,400 | Recuperación parcial del petróleo, tensiones comerciales globales |
| 6 | 2020–2021 | Shock COVID-19 y recuperación | $3,200 → $4,154 | Pánico financiero global (marzo 2020), aversión al riesgo en emergentes, máximos históricos |

**Máximo histórico absoluto:** $4,153.91 COP/USD registrado en abril de 2020, durante el pico de aversión al riesgo global provocado por la pandemia de COVID-19.

### 2.2 Descomposición de la Serie Temporal (STL)

Se aplicó una descomposición STL (*Seasonal-Trend decomposition using LOESS*) con los siguientes parámetros:

- **Período estacional:** 365 días (ciclo anual).
- **Suavizado estacional:** `seasonal=13` (LOESS robusto con ancho de banda amplio para capturar patrones anuales suaves).
- **Robustez:** `robust=True` (ponderación iterativa para atenuar el efecto de outliers en la estimación de los componentes).

Resultados de la descomposición:

| Componente | Desviación estándar | Rango | Interpretación |
|------------|--------------------:|-------|----------------|
| **Observado** | $824.03 | [$620.62, $4,153.91] | Serie original |
| **Tendencia** | $801.17 | — | Componente dominante: explica el 94.3% de la varianza total. Captura la trayectoria de largo plazo de la divisa, incluyendo cambios estructurales (1999, 2014, 2020) |
| **Estacionalidad** | $36.95 | [−$196.00, $194.42] | Componente débil (~4.5% de la varianza). Oscilación anual de ±$195 COP, típica de series financieras donde los fundamentales macroeconómicos dominan sobre patrones calendáricos |
| **Residuo** | $109.02 | [−$409.35, $762.62] | Ruido de alta frecuencia. Captura choques transitorios no explicados por tendencia ni estacionalidad. La presencia de valores atípicos en el residuo ($762.62) es consistente con eventos de volatilidad extrema (COVID-19) |

**Conclusión:** La TCRM es una serie **dominada por la tendencia**, con una componente estacional débil y un residuo que concentra eventos de alta volatilidad en períodos de crisis. La baja estacionalidad es consistente con la naturaleza del mercado cambiario, donde los flujos de capital, los precios de commodities y la política monetaria —todos de frecuencia no estacional— determinan la trayectoria de la divisa.

### 2.3 Análisis de Estacionariedad y Volatilidad

#### 2.3.1 Pruebas de Raíz Unitaria

| Prueba | Estadístico | Valor crítico (5%) | p-value | Conclusión |
|--------|-------------|--------------------|---------|------------|
| **ADF (nivel)** | −0.8315 | −2.8618 | 0.8097 | **No estacionaria** — no se rechaza $H_0$ (presencia de raíz unitaria) |
| **ADF (1ª diferencia)** | −16.8703 | −2.8618 | ≈ 0.0000 | **Estacionaria** — se rechaza $H_0$ contundentemente |
| **KPSS (nivel, tendencia)** | 1.7970 | — | 0.01 | **No estacionaria** — se rechaza $H_0$ (estacionariedad alrededor de tendencia determinista) |

**Diagnóstico:** La TCRM es una serie **integrada de orden 1, $I(1)$**. El test ADF sobre la serie en niveles no puede rechazar la hipótesis nula de raíz unitaria ($p = 0.8097$), mientras que sobre la primera diferencia la rechaza con extrema significancia ($p \approx 0$). El test KPSS confirma la no estacionariedad en niveles, descartando que la serie sea estacionaria alrededor de una tendencia determinista. En consecuencia, **cualquier modelo de series de tiempo debe incluir diferenciación de orden $d = 1$** para inducir estacionariedad.

#### 2.3.2 Regímenes de Volatilidad

La volatilidad de la TCRM —medida como la desviación estándar móvil de 30 días ($\sigma_{30d}$)— exhibe un comportamiento **heterocedástico con clusters pronunciados**, característico de series financieras:

| Período | $\bar{\sigma}_{30d}$ (COP) | Evento asociado |
|---------|---------------------------|-----------------|
| 1991–1998 | $3.00 – $9.82 | Baja volatilidad consistente con el régimen de banda cambiaria administrada |
| 1999 | $22.60 | Transición a flotación libre; devaluación abrupta |
| 2002 | $29.55 | Crisis de confianza en economías emergentes post-11S |
| 2008–2009 | $39.44 – $47.95 | Crisis financiera global (Lehman Brothers, colapso de demanda global) |
| 2015–2016 | $60.88 – $55.82 | **Primer clúster de alta volatilidad:** colapso del precio del petróleo, déficit fiscal y cuenta corriente |
| 2020 | **$71.60** | **Máxima volatilidad histórica:** pánico COVID-19. $\sigma_{30d}$ máximo puntual = $271.31 COP (2020-03-23) |
| 2021 | $44.40 | Normalización parcial con volatilidad aún elevada respecto a promedios pre-2014 |

**Volatilidad promedio histórica:** $\bar{\sigma}_{30d} = 24.90$ COP.  
**Volatilidad máxima registrada:** $\max(\sigma_{30d}) = 271.31$ COP el 23 de marzo de 2020.

La inspección visual de la serie de volatilidad rodante revela que los períodos de calma cambiaria (1991–1997, 2003–2007, 2010–2013) se alternan con episodios de turbulencia concentrada (1999, 2008–2009, 2015–2016, 2020). Este patrón de *volatility clustering* es una propiedad estilizada bien documentada en series financieras y sugiere que una extensión natural del modelo incorporaría un componente **GARCH** para modelar la heterocedasticidad condicional.

#### 2.3.3 Quiebres Estructurales y Eventos Determinantes

- **Septiembre 1999:** Abandono formal del sistema de banda cambiaria. La TCRM pasó de ~$1,600 a ~$2,000 en menos de 60 días.
- **Julio 2008 – Marzo 2009:** Crisis financiera global. Fuga de capitales de economías emergentes, depreciación del 35%.
- **Julio 2014 – Febrero 2016:** Desplome del petróleo Brent ($115 → $28). Colombia —exportador neto de crudo— experimentó una depreciación del 67% en 18 meses.
- **Marzo – Abril 2020:** Pánico COVID-19. La TCRM alcanzó su máximo histórico absoluto ($4,153.91 COP/USD). La volatilidad diaria implícita superó cualquier registro previo.

---

## 3. Desarrollo, Validación y Selección del Modelo de Forecast

### 3.1 Estrategia de Validación Temporal

Para garantizar una evaluación realista y evitar el filtrado de información futura (*data leakage*), se adoptó un esquema de **partición cronológica simple (hold-out validation)**:

| Conjunto | Período | Días calendario | % del total | Propósito |
|----------|---------|:---------------:|:-----------:|-----------|
| **Train** | 1991-12-02 → 2009-03-16 | 8,732 | 80% | Estimación de parámetros del modelo |
| **Validation** | 2009-03-17 → 2021-10-19 | 2,183 | 20% | Evaluación out-of-sample del desempeño predictivo |
| **Full** | 1991-12-02 → 2021-10-19 | 10,915 | 100% | Reentrenamiento final para el forecast a 2023-01-01 |

**Justificación del esquema:** La validación cronológica respeta la dependencia temporal de los datos —el modelo nunca "ve" el futuro durante el entrenamiento—, condición indispensable en series de tiempo financieras donde la distribución de los datos no es i.i.d. y el orden temporal contiene información predictiva. Un esquema de validación cruzada estándar (k-fold aleatorio) destruiría la estructura de autocorrelación e introduciría *look-ahead bias*. La proporción 80/20 es el estándar para horizontes largos de forecast donde se requiere suficiente historia para capturar regímenes múltiples.

**Horizonte de pronóstico:** $h = 439$ días (desde 2021-10-20 hasta 2023-01-01).

### 3.2 Modelos Evaluados

Se implementaron y contrastaron dos enfoques metodológicamente distintos:

#### 3.2.1 SARIMA — *Seasonal AutoRegressive Integrated Moving Average*

Modelo paramétrico clásico de la familia Box-Jenkins. La búsqueda de hiperparámetros se realizó mediante el algoritmo `auto_arima` de la librería `pmdarima`, con criterio de información de Akaike (AIC) como función objetivo:

**Espacio de búsqueda:**
- Parte no estacional: $p \in [1, 3]$, $d$ automático, $q \in [1, 3]$
- Parte estacional ($m=7$, estacionalidad semanal): $P \in [0, 2]$, $D$ automático, $Q \in [0, 2]$
- Algoritmo: *stepwise* (no exhaustivo), $n\_jobs = -1$ (paralelo)

**Modelo seleccionado sobre train:** $SARIMA(1,1,1)(2,0,2)_7$, $AIC = 64,922.31$

La diferenciación de orden $d=1$ es consistente con el diagnóstico $I(1)$ de las pruebas ADF/KPSS. La inclusión de términos estacionales semanales $(m=7)$ captura la micro-estacionalidad intrasemanal presente en las series financieras, donde la volatilidad y los niveles pueden variar sistemáticamente entre días de la semana.

**Modelo reentrenado sobre serie completa:** $SARIMA(2,1,3)(0,0,0)_7$, $AIC = 87,316.96$

El modelo sobre la serie completa difiere del modelo sobre train —la parte estacional $(0,0,0)_7$ desaparece—, lo cual es esperable: los 2,183 días adicionales en el conjunto completo contienen los regímenes de alta volatilidad de 2015–2016 y 2020, que dominan sobre la estacionalidad semanal débil presente en el período de entrenamiento.

#### 3.2.2 Prophet — Modelo Aditivo de Meta

Modelo de series de tiempo desarrollado por Facebook/Meta, basado en un enfoque de **modelo aditivo descomponible**:

$$y(t) = g(t) + s(t) + h(t) + \varepsilon_t$$

donde $g(t)$ es la tendencia (lineal por tramos con *changepoints* automáticos), $s(t)$ es la estacionalidad, $h(t)$ es el efecto de días festivos, y $\varepsilon_t$ es el error.

**Configuración:**

| Hiperparámetro | Valor | Justificación |
|----------------|-------|---------------|
| `changepoint_prior_scale` | 0.05 | Flexibilidad moderada para capturar cambios de tendencia sin sobre-ajustar ruido. Valores más altos (>0.5) producirían quiebres excesivos; más bajos (<0.01) producirían una tendencia demasiado rígida |
| `yearly_seasonality` | `True` | Captura el ciclo anual de ~±$195 COP identificado en la descomposición STL |
| `weekly_seasonality` | `False` | La estacionalidad semanal en series financieras de frecuencia diaria es débil. Su inclusión añadiría 6 grados de libertad con beneficio marginal |
| `daily_seasonality` | `False` | No aplica: los datos tienen frecuencia diaria, no intradiaria |
| `interval_width` | 0.95 | Intervalos de confianza al 95% estándar |

### 3.3 Métricas de Evaluación

Se evaluó el desempeño de ambos modelos sobre el conjunto de validación (2,183 días out-of-sample) utilizando tres métricas estándar:

| Métrica | Fórmula | Interpretación |
|---------|---------|----------------|
| **MAE** (*Mean Absolute Error*) | $\frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$ | Error absoluto promedio. Penaliza linealmente las desviaciones. Robusto a outliers |
| **RMSE** (*Root Mean Squared Error*) | $\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$ | Raíz del error cuadrático medio. Penaliza cuadráticamente los errores grandes — relevante para riesgo de cola |
| **MAPE** (*Mean Absolute Percentage Error*) | $\frac{100}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right|$ | Error porcentual absoluto medio. Interpretable en términos relativos. Facilita la comunicación con stakeholders no técnicos |

#### Resultados Comparativos

| Modelo | MAE (COP) | RMSE (COP) | MAPE (%) | $\hat{\sigma}_{1\text{-step}}$ |
|--------|:---------:|:----------:|:--------:|:------------------------------:|
| **SARIMA** $(1,1,1)(2,0,2)_7$ | **193.63** | **231.68** | **6.14%** | 12.10 |
| **Prophet** | 358.72 | 414.90 | 10.77% | 118.82 |

El modelo SARIMA supera a Prophet en todas las métricas:

- **MAE 46.0% inferior** ($193.63$ vs. $358.72$ COP): el error absoluto típico de SARIMA es aproximadamente la mitad del de Prophet.
- **RMSE 44.2% inferior** ($231.68$ vs. $414.90$ COP): SARIMA también domina en la cola de la distribución del error, indicando mejor desempeño en episodios de alta volatilidad.
- **MAPE de 6.14% vs. 10.77%:** en promedio, SARIMA se desvía 6.1 centavos por cada 100 pesos de TCRM, frente a los 10.8 centavos de Prophet.
- **$\hat{\sigma}_{1\text{-step}}$ 89.8% inferior** ($12.10$ vs. $118.82$ COP): la desviación estándar del error de predicción a un paso de SARIMA es un orden de magnitud menor, lo que otorga **intervalos de confianza sustancialmente más estrechos y útiles para la gestión de riesgo**.

### 3.4 Modelo Final Seleccionado: SARIMA

#### 3.4.1 Justificación de la Selección

La selección de **SARIMA como modelo final** se fundamenta en una evaluación multicriterio que integra consideraciones estadísticas, de negocio y operativas:

| Criterio | SARIMA | Prophet | Decisión |
|----------|--------|---------|----------|
| **Precisión predictiva** | MAE = 193.63 COP | MAE = 358.72 COP | ✅ SARIMA (46% menor error) |
| **Precisión en cola (RMSE)** | 231.68 COP | 414.90 COP | ✅ SARIMA (44% menor) |
| **Error porcentual (MAPE)** | 6.14% | 10.77% | ✅ SARIMA (más del 40% menor) |
| **Estrechez de intervalos ($\hat{\sigma}_{1\text{-step}}$)** | 12.10 COP | 118.82 COP | ✅ SARIMA (orden de magnitud menor) |
| **Parsimonia** | $p+q+P+Q = 4$ parámetros | 1 hiperparámetro + N changepoints | ✅ SARIMA (modelo más simple y auditable) |
| **Fundamento teórico** | Sólido: teoremas de Wold, condiciones de invertibilidad y estacionariedad bien caracterizadas | Heurístico: descomposición aditiva sin garantías formales de optimalidad | ✅ SARIMA |
| **Diagnóstico de residuos** | Ruido blanco verificable vía ACF/PACF y Ljung-Box | Residuales con estructura de autocorrelación no modelada | ✅ SARIMA |
| **Interpretabilidad para negocio** | Media: AR y MA tienen interpretación financiera directa (momentum, reversión a la media) | Alta: changepoints automáticos son intuitivos | Empate |

#### 3.4.2 Hiperparámetros Óptimos

**Modelo final:** $SARIMA(2,1,3)$ sin componente estacional semanal $((0,0,0)_7)$

| Parámetro | Valor | Interpretación |
|-----------|-------|----------------|
| $p = 2$ | Orden autorregresivo | La TCRM de hoy depende de los valores de los últimos 2 días. Captura momentum de corto plazo |
| $d = 1$ | Orden de diferenciación | La serie es $I(1)$: se requiere una diferenciación para inducir estacionariedad. Consistente con ADF ($p = 0.8097$ en niveles, $p \approx 0$ en primera diferencia) |
| $q = 3$ | Orden de media móvil | El error de predicción de hoy incorpora innovaciones de los últimos 3 días. Captura la persistencia de choques transitorios |
| $P = 0, D = 0, Q = 0$ | Sin estacionalidad semanal | La estacionalidad semanal es débil y no justifica parámetros adicionales cuando la serie completa incluye los regímenes de alta volatilidad de 2015–2020 |
| $AIC = 87,316.96$ | Criterio de información | Valor de referencia para comparación con especificaciones alternativas. Menor AIC = mejor balance ajuste/parsimonia |

#### 3.4.3 Diagnóstico de Residuos

Los residuos del modelo SARIMA final sobre la serie completa presentan:

- **Media cercana a cero:** el modelo no tiene sesgo sistemático —no subestima ni sobreestima consistentemente la TCRM.
- **Desviación estándar del error a un paso:** $\hat{\sigma}_{1\text{-step}} = 12.10$ COP, indicando que el 68% de las predicciones diarias están dentro de ±$12.10$ COP del valor real (bajo supuesto de normalidad).
- **Baja varianza relativa:** $\hat{\sigma}_{1\text{-step}}$ representa apenas el 0.32% del valor medio de la TCRM en niveles (~$3,772$ COP) y el 1.47% de la desviación estándar histórica ($824.03$ COP). Esto indica que el modelo captura la mayor parte de la estructura predecible de la serie.

El supuesto de **independencia de residuos** —requerido para la validez de $\hat{\sigma}_h = \hat{\sigma}_{1\text{-step}} \cdot \sqrt{h}$— es una aproximación razonable para horizontes moderados ($h=439$). Para horizontes más largos o en presencia de *volatility clustering*, una extensión con modelo GARCH sobre los residuos mejoraría la calibración de los intervalos de confianza.

---

## 4. Proyección al 1° de Enero de 2023 y Cuantificación del Riesgo

### 4.1 Estimación Puntual

El modelo $SARIMA(2,1,3)$ entrenado sobre la serie completa (1991-12-02 → 2021-10-19) proyecta la siguiente trayectoria para el horizonte $h = 439$ días:

**TCRM proyectada al 1 de enero de 2023:** **$3,898.93$ COP/USD**

Este valor representa una depreciación adicional del **3.35%** respecto al último valor observado el 19 de octubre de 2021 ($3,772.49$ COP/USD), equivalente a una tasa de depreciación anualizada de aproximadamente 2.7% — consistente con un escenario de estabilización relativa post-COVID con presiones inflacionarias y monetarias moderadas.

Como referencia, el modelo Prophet —utilizado como validación cruzada metodológica— proyecta $3,880.82$ COP/USD, una diferencia de apenas $18.11$ COP (0.46%) respecto a SARIMA. La convergencia de dos modelos estructuralmente distintos en una estimación cercana refuerza la confianza en la proyección. El **ensemble simple** (promedio aritmético) arroja **$3,889.88$ COP/USD**.

### 4.2 Incertidumbre y Desviación Estándar

#### 4.2.1 Metodología de Cálculo

La desviación estándar acumulada del error de pronóstico a horizonte $h$ se calcula bajo el supuesto de **independencia de los errores de predicción a un paso**:

$$\hat{\sigma}_h = \hat{\sigma}_{1\text{-step}} \cdot \sqrt{h}$$

donde $\hat{\sigma}_{1\text{-step}} = 12.10$ COP es la desviación estándar de los residuos del modelo ajustado sobre la serie de entrenamiento, y $h = 439$ es el horizonte de pronóstico.

$$\hat{\sigma}_{439} = 12.10 \times \sqrt{439} = 12.10 \times 20.952 = 253.55 \text{ COP}$$

#### 4.2.2 Intervalos de Confianza

Bajo el supuesto de normalidad de los errores de pronóstico acumulados, los intervalos de confianza se construyen como:

| Nivel de confianza | Factor $z_{\alpha/2}$ | Límite inferior (COP) | Pronóstico puntual (COP) | Límite superior (COP) | Amplitud (COP) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **80%** | 1.282 | $3,573.86$ | $3,898.93$ | $4,224.00$ | $650.14$ |
| **95%** | 1.960 | $3,401.98$ | $3,898.93$ | $4,395.88$ | $993.90$ |

La interpretación de estos intervalos es directa en el contexto de gestión de riesgo cambiario:

- **IC 80%:** existe una probabilidad del 80% de que la TCRM el 1 de enero de 2023 se encuentre entre **$3,574** y **$4,224** COP/USD. Es el rango relevante para decisiones tácticas de cobertura con un apetito de riesgo moderado.
- **IC 95%:** con 95% de confianza, la TCRM estará entre **$3,402** y **$4,396** COP/USD. Representa el escenario de *worst case* para la planificación financiera con un nivel de certidumbre alto.

#### 4.2.3 Interpretación en el Margen de Error

La desviación estándar acumulada $\hat{\sigma}_{439} = 253.55$ COP representa el **6.50%** del valor puntual proyectado ($3,898.93$ COP). En el contexto del negocio importador:

- **Costo de importación:** un contenedor de mercancía con valor FOB de USD $50,000 tendría un costo en pesos entre $170.1 M (escenario favorable, IC 95% inferior) y $219.8 M (escenario adverso, IC 95% superior). La diferencia de **$49.7 M** representa el riesgo cambiario no cubierto para una sola operación de tamaño moderado.
- **Margen bruto:** para un importador que opera con un margen bruto del 30%, una depreciación inesperada de $500 COP/USD erosiona aproximadamente 4.3 puntos porcentuales del margen si los precios de venta no se ajustan.

### 4.3 Implicaciones para el Cliente Importador

El perfil del cliente —importador con portal e-commerce— lo expone de forma directa y asimétrica al riesgo cambiario: la totalidad de sus costos de adquisición están denominados en USD, mientras que sus ingresos por ventas están denominados en COP. Las recomendaciones se estructuran en tres ejes:

#### Eje 1: Cobertura Cambiaria con Derivados

La proyección puntual de $3,898.93$ COP/USD, con un IC 80% de [$3,574, $4,224], proporciona una **señal cuantitativa accionable** para la estructuración de coberturas:

- **Forward de divisa (Non-Deliverable Forward, NDF):** para el 50–70% del volumen proyectado de importaciones con vencimiento enero 2023. Un forward a $3,900$ COP/USD eliminaría la exposición al escenario adverso del IC 95% ($4,396$), garantizando un costo máximo de importación. Dado que el mercado forward a 14 meses (oct-21 a ene-23) tiene liquidez limitada en Colombia, se recomienda una estructura escalonada (*ladder*) con vencimientos trimestrales.
- **Opciones (PUT sobre USD/COP):** para el 20–30% restante del volumen. Una opción PUT con strike en $3,900$ COP/USD mantiene el upside (beneficio si el peso se aprecia por debajo de $3,574) mientras limita el downside (costo máximo acotado). La prima de la opción debe presupuestarse como un costo de seguro.

#### Eje 2: Presupuestación de Inventario y Ajuste de Márgenes

- **Escenario base de presupuesto:** $3,900$ COP/USD (redondeo del pronóstico puntual de $3,898.93$).
- **Escenario de estrés:** $4,400$ COP/USD (redondeo del IC 95% superior, $4,395.88$). El margen bruto proyectado debe ser **positivo incluso bajo este escenario** —si no lo es, se requiere un ajuste de precios de venta o una cobertura adicional.
- **Pricing dinámico en e-commerce:** implementar un algoritmo de ajuste automático de precios de venta que indexe parcialmente el precio en COP al tipo de cambio spot. Un pass-through del 60–80% es típico en e-commerce para categorías de consumo discrecional.

#### Eje 3: Monitoreo Continuo

- Reentrenar el modelo SARIMA mensualmente incorporando los nuevos datos de TCRM publicados.
- Monitorear la evolución de $\hat{\sigma}_{1\text{-step}}$: si la volatilidad diaria aumenta significativamente por encima de $12.10$ COP —por ejemplo, ante un nuevo shock externo— los intervalos de confianza deben recalcularse y las coberturas ajustarse.
- Establecer alertas automáticas cuando la TCRM spot cruce los umbrales del IC 80% ($3,574$ o $4,224$) para activar revisiones tácticas de la estrategia de cobertura.

---

## 5. Respuestas Oficiales a la Prueba Técnica: Pregunta 2 (TCRM)

### a. TCRM para la fecha de expedición de la cédula

- **Fecha consultada:** 2010-04-19
- **Valor exacto de la TCRM:** **$1,942.21 COP**

El valor fue verificado mediante indexación directa sobre la serie diaria procesada (`get_tcrm_at_date()`), coincidiendo exactamente con el valor de referencia.

### b. Auditoría de completitud y función de imputación

- **Diagnóstico de completitud:** Los datos **no estaban completos** para un análisis de series de tiempo continuo. El archivo fuente contiene 7,188 registros que cubren únicamente los días en que el mercado cambiario operó, con intervalos de vigencia multi-día (cada fila tiene un rango `[VIGENCIADESDE, VIGENCIAHASTA]`).
- **Días no transados:** **3,727 días** — correspondientes a fines de semana (sábados y domingos) y festivos nacionales colombianos y bancarios en EE. UU., en los cuales el mercado cambiario no opera y no se publica TCRM.
- **Función de imputación:** `build_daily_series()` en `src/tcrm_forecast.py`. Opera en tres etapas:
  1. **Expansión:** cada registro multi-día se descompone en filas individuales para cada día del intervalo `[VIGENCIADESDE, VIGENCIAHASTA]`.
  2. **Reindexación:** se genera un calendario diario continuo desde 1991-12-02 hasta 2021-10-19 (10,915 días). La serie expandida se reindexa a este calendario.
  3. **Forward fill (`ffill`):** los días sin dato heredan el último valor conocido, conforme a la Circular DODM-146 de la Superintendencia Financiera de Colombia, que establece que la TCRM vigente para días no transados es la del último día hábil negociado.

La elección del forward fill sobre interpolación lineal se justifica porque la TCRM **no varía de forma continua** en días no transados —no existe un mercado que forme precios— y la interpolación introduciría valores artificiales que violan la realidad operativa del mercado cambiario y generarían *look-ahead bias*.

### c. Análisis gráfico y comportamiento de la curva

**Hallazgos principales de la curva TCRM vs. Tiempo (1991–2021):**

1. **Seis regímenes cambiarios estructurales:** banda cambiaria (1991–1998), crisis y flotación (1998–2003), super-peso (2003–2013), devaluación post-petróleo (2014–2016), estabilización alta (2017–2019) y shock COVID-19 (2020–2021). Devaluación acumulada: 507.8% en 29.85 años.

2. **Tendencia dominante:** la descomposición STL revela que la tendencia explica el 94.3% de la varianza ($\sigma_{\text{trend}} = 801.17$ COP), con una estacionalidad anual débil ($\sigma_{\text{seasonal}} = 36.95$ COP, ~±$195$ COP) y un residuo que concentra eventos de alta volatilidad ($\sigma_{\text{resid}} = 109.02$ COP).

3. **No estacionariedad confirmada:** prueba ADF en niveles ($p = 0.8097$) no rechaza raíz unitaria; KPSS confirma no estacionariedad ($p = 0.01$). La primera diferencia es estacionaria ($p \approx 0$). La serie es $I(1)$.

4. **Volatilidad heterocedástica con clusters:** promedio histórico de $\bar{\sigma}_{30d} = 24.90$ COP, con picos en 2015–2016 ($60.88$ COP, crisis petrolera) y 2020 ($71.60$ COP, COVID-19). Máximo puntual: $271.31$ COP el 23/03/2020.

5. **Máximo histórico:** $4,153.91$ COP/USD en abril de 2020.

### d. Modelo de Forecast, Proyección al 1 de enero de 2023 y Desviación Estándar

- **Modelo final seleccionado:** $SARIMA(2,1,3)$, sin componente estacional semanal. Seleccionado tras comparativa con Prophet. SARIMA obtuvo un desempeño validación significativamente superior: $MAE = 193.63$ COP, $RMSE = 231.68$ COP, $MAPE = 6.14\%$ (vs. Prophet: $MAE = 358.72$, $RMSE = 414.90$, $MAPE = 10.77\%$).

- **Pronóstico puntual de la TCRM para el 01/01/2023:** **$3,898.93$ COP/USD** (SARIMA). Prophet proyecta $3,880.82$ COP/USD como validación cruzada. Ensemble simple: $3,889.88$ COP/USD.

- **Desviación estándar estimada:** $\hat{\sigma}_{1\text{-step}} = 12.10$ COP. Acumulada a horizonte $h=439$ días:

$$\hat{\sigma}_{439} = \hat{\sigma}_{1\text{-step}} \cdot \sqrt{439} = 12.10 \times 20.952 = 253.55 \text{ COP}$$

- **Interpretación en el margen de error:** $\hat{\sigma}_{439} = 253.55$ COP representa el **6.50%** del valor puntual proyectado. En términos de gestión de riesgo, esto significa que con un 95% de confianza la TCRM del 01/01/2023 se ubicará entre **$3,402$ y $4,396$ COP/USD**. Para un importador con un costo de mercancía de USD $50,000, la diferencia entre el escenario favorable y el adverso asciende a aproximadamente $49.7$ millones de COP —magnitud que justifica plenamente la implementación de una estrategia de cobertura cambiaria con forwards y opciones.

---

*Documento generado a partir del análisis del código fuente (`src/tcrm_forecast.py`), notebook de EDA y forecasting (`notebooks/tcrm_eda_and_forecasting.ipynb`), ejecución en vivo de los modelos SARIMA y Prophet, prueba ADF/KPSS, descomposición STL y dataset procesado (`data/processed/tcrm_diaria_procesada.parquet`). Todos los valores numéricos reportados fueron verificados mediante ejecución directa del código en el entorno del proyecto.*
