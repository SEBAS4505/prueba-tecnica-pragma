# Prueba Técnica — Analista de Datos

Proyecto de análisis de datos con segmentación de comercios para banca digital, forecasting de TCRM y simulación financiera.

## Estructura del proyecto

```
├── src/
│   ├── etl.py                  # Pipeline ETL: carga, limpieza, RFM y exportación
│   ├── segmentation.py         # Segmentación K-Means (deprecado — se usó K-Medoids)
│   ├── gen_kmedoids_nb.py      # Generador del notebook de segmentación K-Medoids
│   ├── tcrm_forecast.py        # Pipeline TCRM: carga, limpieza, forward-fill y exportación
│   ├── simulacion_financiera.py# Waterfall financiero: 3 escenarios de crecimiento
│   ├── tablas_bi_gen.py        # Generación de tablas optimizadas para Power BI
│   ├── utils.py                # Utilidades compartidas
│   └── main.py                 # Script de ejemplo (PyCharm template)
├── notebooks/
│   ├── segmentation_evaluation_k_medoids_var3.ipynb  # K-Medoids — variante 3
│   ├── segmentation_evaluation_k_medoids_var6.ipynb  # K-Medoids — variante 6
│   ├── segmentation_evaluation_k_medoids.ipynb       # K-Medoids — notebook principal
│   └── tcrm_eda_and_forecasting.ipynb                # EDA y forecasting TCRM
├── data/
│   ├── raw/                    # Datos de entrada (CSV, Excel)
│   └── processed/              # Datos procesados y modelos serializados
├── documentacion_fase1_segmentacion_kmedoids.md
├── documentacion_fase2_tcrm_forecast.md
├── reporte_segmentacion_comercios.pbix
└── requirements.txt
```

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

**Dependencias principales:** `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `prophet`, `pmdarima`, `matplotlib`, `seaborn`, `openpyxl`, `jupyter`.

## Componentes

### 1. Segmentación de comercios (K-Medoids)

Segmentación RFM de 3 015 comercios activos (últimos 6 meses, corte: 2021-05-01) usando K-Medoids (PAM) con K=4.

**Segmentos:**
- **Champions / VIP** — alta frecuencia y valor
- **Core / Crecimiento** — volumen moderado, potencial de expansión
- **En Riesgo / Churn** — baja recencia, riesgo de abandono
- **Ocasional / Ticket Alto** — baja frecuencia pero ticket elevado

**Ejecución:**
```bash
python src/etl.py                              # ETL + RFM
python src/gen_kmedoids_nb.py                  # genera el notebook
# Luego ejecutar celdas en notebooks/segmentation_evaluation_k_medoids.ipynb
python src/tablas_bi_gen.py                    # tablas para Power BI
```

**Métricas del modelo (K=4):** Silhouette Score ~0.47, Davies-Bouldin ~0.93, ARI medio > 0.99 (alta estabilidad).

### 2. Análisis y forecasting TCRM

Serie diaria continua desde 1991-12-02 hasta 2021-10-19 (10 915 días). Forward-fill normativo para 3 727 días no transados (fines de semana y festivos colombianos).

```bash
python src/tcrm_forecast.py
```

Forecasting con Prophet y ARIMA en `notebooks/tcrm_eda_and_forecasting.ipynb`.

### 3. Simulación financiera (Waterfall)

Modelo de crecimiento para alcanzar +50% en comisiones (~COP $330M), con tres escenarios (Conservador, Intermedio, Acelerado) sobre dos pilares:
- **Cartera actual** — retención y crecimiento por segmento
- **Adquisición neta** — captación de comercios Look-Alike

```bash
python src/simulacion_financiera.py
```

### 4. Power BI

Tablero de segmentación en `reporte_segmentacion_comercios.pbix`, alimentado por los archivos generados en `data/processed/`.

## Ejecución completa

```bash
# 1. ETL y segmentación
python src/etl.py
python src/tablas_bi_gen.py

# 2. TCRM
python src/tcrm_forecast.py

# 3. Simulación financiera
python src/simulacion_financiera.py
```
