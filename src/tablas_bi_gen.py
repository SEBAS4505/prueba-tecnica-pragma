"""
Generación de tablas optimizadas para Power BI
==============================================
Genera dos entregables a partir de los datos procesados por el ETL
y la segmentación K-Medoids, evitando cálculos pesados y relaciones
Many-to-Many en Power BI.

Entregables:
    1. transacciones_con_segmento.csv — JOIN transacciones + segmento
    2. comercios_final_k_medoids.csv — enriquecido con columnas pre-calculadas

Uso:
    python src/tablas_bi_gen.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Rutas base
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

TRANSACCIONES_FILE = RAW_DIR / "DataTransacciones_PruebasIngreso_Transacciones.csv"
K_MEDOIDS_FILE = PROCESSED_DIR / "comercios_final_k_medoids.csv"

OUTPUT_TRANS_SEGMENTO = PROCESSED_DIR / "transacciones_con_segmento.csv"


# ---------------------------------------------------------------------------
# Helpers (misma lógica que etl.py)
# ---------------------------------------------------------------------------

def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza los nombres de columnas a snake_case limpio.

    Convierte a minúsculas, reemplaza espacios y caracteres especiales
    por guiones bajos, y elimina guiones bajos dobles o al final.
    """
    new_cols = []
    for col in df.columns:
        normalized = (
            col.strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
            .replace(".", "")
        )
        normalized = re.sub(r"_+", "_", normalized).strip("_")
        new_cols.append(normalized)
    df.columns = new_cols
    return df


def _parse_mes_anio(val: str) -> pd.Timestamp | None:
    """Parsea 'mes-anio' (ej. '3-2021') a Timestamp del primer día del mes."""
    try:
        parts = str(val).strip().split("-")
        month, year = int(parts[0]), int(parts[1])
        return pd.Timestamp(year=year, month=month, day=1)
    except (ValueError, IndexError):
        return pd.NaT


# ---------------------------------------------------------------------------
# 1. Generar transacciones_con_segmento.csv
# ---------------------------------------------------------------------------

def generate_transactions_with_segment() -> pd.DataFrame:
    """Genera la tabla de transacciones con el segmento asignado por K-Medoids.

    Realiza un JOIN entre las transacciones originales y el resultado del
    modelo K-Medoids por ``id_comercio``, aplicando la misma limpieza que
    :func:`etl.clean_transacciones`.

    Operaciones aplicadas:
        - Conversión de ``"mes-anio"`` a ``datetime`` (primer día del mes).
        - Conversión de ``cantidad_transacciones`` a entero.
        - Filtrado de los últimos 6 meses tomando la fecha máxima del
          dataset como punto de corte.

    Returns:
        DataFrame con columnas: id_comercio, fecha_transaccion,
        cantidad_transacciones, cluster, segmento.
    """
    print("[BI-GEN] Generando transacciones_con_segmento.csv ...")

    # --- Cargar transacciones crudas ---
    df_trans = pd.read_csv(
        TRANSACCIONES_FILE,
        sep=";",
        encoding="latin-1",
        dtype=str,
    )
    df_trans = _normalize_column_names(df_trans)

    # --- Limpiar transacciones (misma lógica que etl.clean_transacciones) ---
    df_trans["fecha_transaccion"] = df_trans["mes_anio_transaccion"].apply(
        _parse_mes_anio
    )
    df_trans["cantidad_transacciones"] = (
        pd.to_numeric(df_trans["cantidad_transacciones"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    # Filtrar últimos 6 meses desde la fecha máxima del dataset
    max_date = df_trans["fecha_transaccion"].max()
    cutoff_date = max_date - pd.DateOffset(months=6) + pd.DateOffset(days=1)

    before = len(df_trans)
    df_trans = df_trans[df_trans["fecha_transaccion"] >= cutoff_date].reset_index(
        drop=True
    )
    filtered_out = before - len(df_trans)

    print(f"  [INFO] Fecha máxima del dataset: {max_date.strftime('%Y-%m-%d')}")
    print(f"  [INFO] Corte 6 meses desde: {cutoff_date.strftime('%Y-%m-%d')}")
    if filtered_out > 0:
        print(f"  [INFO] {filtered_out:,} filas excluidas fuera del período.")

    # --- Cargar segmentos desde K-Medoids ---
    df_kmedoids = pd.read_csv(K_MEDOIDS_FILE)
    df_segmentos = df_kmedoids[["id_comercio", "cluster", "segmento"]]

    # --- Merge: JOIN entre transacciones y segmentos ---
    df_result = df_trans.merge(df_segmentos, on="id_comercio", how="left")

    # Verificar comercios sin match
    n_sin_match = df_result["cluster"].isna().sum()
    if n_sin_match > 0:
        print(
            f"  [WARN] {n_sin_match:,} filas de transacciones sin segmento "
            f"asignado (comercios no encontrados en K-Medoids)."
        )

    # --- Seleccionar y ordenar columnas de salida ---
    df_result = df_result[
        [
            "id_comercio",
            "fecha_transaccion",
            "cantidad_transacciones",
            "cluster",
            "segmento",
        ]
    ]

    # --- Exportar ---
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_result.to_csv(OUTPUT_TRANS_SEGMENTO, index=False, encoding="utf-8")

    size_kb = OUTPUT_TRANS_SEGMENTO.stat().st_size / 1024
    print(f"  [OK] Exportado: {OUTPUT_TRANS_SEGMENTO.name}")
    print(f"       Filas: {len(df_result):,} | Columnas: {len(df_result.columns)} | "
          f"Tamaño: {size_kb:.1f} KB")

    return df_result


# ---------------------------------------------------------------------------
# 2. Enriquecer comercios_final_k_medoids.csv
# ---------------------------------------------------------------------------

def enrich_comercios_k_medoids() -> pd.DataFrame:
    """Añade columnas pre-calculadas a ``comercios_final_k_medoids.csv``.

    Columnas añadidas/modificadas:

    - ``fecha_creacion``: transformada a formato ``'YYYY-MM'``.
    - ``dias_para_activacion``: días entre ``fecha_creacion`` y
      ``fecha_primera_transaccion``.
    - ``promedio_venta_mensual_historico``:
      ``monto_ventas_acumulado / active_months`` (redondeado a 2 decimales).

    Returns:
        DataFrame enriquecido, con las mismas filas que el original
        y 2 columnas nuevas adicionales.
    """
    print("[BI-GEN] Enriqueciendo comercios_final_k_medoids.csv ...")

    df = pd.read_csv(K_MEDOIDS_FILE)

    # --- Guardar fecha original para el cálculo de días ---
    fecha_creacion_dt = pd.to_datetime(df["fecha_creacion"])
    fecha_primera_dt = pd.to_datetime(df["fecha_primera_transaccion"])

    # --- Transformar fecha_creacion a formato YYYY-MM ---
    df["fecha_creacion"] = fecha_creacion_dt.dt.strftime("%Y-%m")

    # --- Calcular días para activación ---
    df["dias_para_activacion"] = (fecha_primera_dt - fecha_creacion_dt).dt.days

    # --- Calcular promedio de venta mensual histórico ---
    df["promedio_venta_mensual_historico"] = np.where(
        df["active_months"] > 0,
        df["monto_ventas_acumulado"] / df["active_months"],
        np.nan,
    )
    df["promedio_venta_mensual_historico"] = (
        df["promedio_venta_mensual_historico"].round(2)
    )

    # --- Exportar (sobrescribe el archivo original) ---
    df.to_csv(K_MEDOIDS_FILE, index=False, encoding="utf-8")

    size_kb = K_MEDOIDS_FILE.stat().st_size / 1024
    print(f"  [OK] Sobreescrito: {K_MEDOIDS_FILE.name}")
    print(f"       Filas: {len(df):,} | Columnas: {len(df.columns)} | "
          f"Tamaño: {size_kb:.1f} KB")

    # --- Mostrar estadísticas de las nuevas columnas ---
    print(f"\n  -- Resumen de nuevas columnas --")
    print(f"  dias_para_activacion: "
          f"min={df['dias_para_activacion'].min():.0f}, "
          f"media={df['dias_para_activacion'].mean():.1f}, "
          f"max={df['dias_para_activacion'].max():.0f}")
    print(f"  promedio_venta_mensual_historico: "
          f"min=${df['promedio_venta_mensual_historico'].min():,.0f}, "
          f"media=${df['promedio_venta_mensual_historico'].mean():,.0f}, "
          f"max=${df['promedio_venta_mensual_historico'].max():,.0f}")

    return df


# ---------------------------------------------------------------------------
# Orquestador
# ---------------------------------------------------------------------------

def run() -> None:
    """Ejecuta la generación completa de tablas para Power BI."""
    print("=" * 60)
    print("  GENERACIÓN DE TABLAS PARA POWER BI")
    print("=" * 60)

    generate_transactions_with_segment()
    print()
    enrich_comercios_k_medoids()

    print()
    print("=" * 60)
    print("  [BI-GEN] Tablas generadas exitosamente.")
    print("=" * 60)


if __name__ == "__main__":
    run()
