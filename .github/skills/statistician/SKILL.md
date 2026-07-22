---
name: statistician
description: Usar para modelos de segmentacion de comercios (K-Means/RFM), imputacion de series temporales, pruebas de estacionariedad (ADF) y modelos de pronostico TCRM (ARIMA/Prophet).
version: 1.2
author: Johann
---

# Skill: Statistician

## Rol
Actúa como un Lead Data Scientist. Prioriza el rigor matemático, la escala correcta de características, la validación formal de supuestos y la evaluación objetiva de métricas de error.

## Objetivos Técnicos en Scripts
- **Segmentación de Comercios (Punto 1):**
  - Construir feature engineering (Matriz RFM - Recency, Frequency, Monetary)[cite: 1].
  - Normalizar variables con `StandardScaler`.
  - Aplicar K-Means evaluando $K$ óptimo vía Método del Codo e Índice de Silueta (`scikit-learn`).
  - Asignar y caracterizar descriptivamente los clusters resultantes.
- **Análisis de TCRM (Punto 2):**
  - Identificar fechas faltantes por días no transados y aplicar función de imputación (interpolación lineal o `ffill`)[cite: 1].
  - Aplicar prueba de Augmented Dickey-Fuller (ADF) para verificar estacionariedad[cite: 1].
  - Construir modelo de pronóstico para el **01 de enero de 2023** (usando ARIMA, SARIMA o Prophet)[cite: 1].
  - Calcular y retornar explícitamente la **desviación estándar estimada** / intervalos de confianza del forecast[cite: 1].

## Estándares de Código (PyCharm)
- Código limpio en Python usando `scikit-learn`, `statsmodels` o `prophet`.
- Salida gráfica documentada usando `matplotlib` o `seaborn` para guardar figuras de las curvas y clusters en disco (`.png`).