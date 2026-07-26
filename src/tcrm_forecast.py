"""
TCRM Forecast Pipeline — prueba-tecnica-pragma
==============================================
Módulo para ingestión, limpieza, imputación de días no transados y
exportación de la Tasa de Cambio Representativa del Mercado (TCRM)
colombiano, siguiendo la reglamentación de la Superintendencia
Financiera de Colombia.

Genera una serie de tiempo diaria continua desde 1991-12-02 hasta
2021-10-19, con forward-fill para los 3,727 días no transados
(fines de semana y festivos).

Uso:
    python src/tcrm_forecast.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas base
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

TCRM_FILE = RAW_DIR / "Tasa_de_Cambio_Representativa_del_Mercado_Historico.xlsx"
OUTPUT_FILE = PROCESSED_DIR / "tcrm_diaria_procesada.parquet"

# Ventana completa esperada de la serie TCRM
_START_DATE: str = "1991-12-02"
_END_DATE: str = "2021-10-19"
_TOTAL_CALENDAR_DAYS: int = 10_915


# ===========================================================================
# Helpers privados
# ===========================================================================

def _parse_colombian_number(value: str) -> float:
    """Convierte un número con formato colombiano (coma como separador de miles,
    punto decimal) a ``float``.

    Args:
        value: Cadena como ``"2,850.98"`` o ``"1850.50"``.

    Returns:
        Valor numérico como :class:`float`, o ``numpy.nan`` si no es convertible.

    Example:
        >>> _parse_colombian_number("2,850.98")
        2850.98
        >>> _parse_colombian_number("1850")
        1850.0
    """
    if pd.isna(value):
        return np.nan
    cleaned = str(value).strip().replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def _generate_date_range(start: str, end: str, freq: str = "D") -> pd.DatetimeIndex:
    """Genera un rango continuo de fechas a frecuencia diaria.

    Args:
        start: Fecha de inicio en formato ``"YYYY-MM-DD"``.
        end: Fecha de fin en formato ``"YYYY-MM-DD"`` (inclusive).
        freq: Frecuencia de pandas (por defecto ``"D"``).

    Returns:
        :class:`~pandas.DatetimeIndex` con todas las fechas del rango.
    """
    return pd.date_range(start=start, end=end, freq=freq)


# ===========================================================================
# Funciones principales del pipeline TCRM
# ===========================================================================

def load_tcrm_raw() -> pd.DataFrame:
    """Carga el archivo Excel histórico de la TCRM.

    Returns:
        DataFrame crudo con columnas ``VALOR``, ``UNIDAD``, ``VIGENCIADESDE``,
        ``VIGENCIAHASTA``.

    Raises:
        FileNotFoundError: Si el archivo Excel no existe en ``RAW_DIR``.
    """
    print("[TCRM] Cargando datos crudos desde Excel...")

    if not TCRM_FILE.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {TCRM_FILE}")

    df = pd.read_excel(TCRM_FILE)

    print(f"  [OK] Cargadas {len(df):,} filas | {len(df.columns)} columnas")
    return df


def clean_tcrm(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia y normaliza el DataFrame de TCRM.

    Operaciones aplicadas:

    - Conversión de ``VALOR`` a ``float`` eliminando separadores de miles.
    - Conversión de ``VIGENCIADESDE`` y ``VIGENCIAHASTA`` a ``datetime``.
    - Ordenamiento cronológico ascendente.
    - Reseteo de índice.

    Args:
        df: DataFrame crudo de TCRM (salida de :func:`load_tcrm_raw`).

    Returns:
        DataFrame limpio y ordenado con ``VALOR`` como ``float64`` y fechas
        como ``datetime64``.
    """
    print("[TCRM] Limpiando y normalizando datos...")

    df = df.copy()

    # --- Limpiar VALOR: remover comas de miles y convertir a float ---
    df["VALOR"] = df["VALOR"].apply(_parse_colombian_number)
    n_nulls = df["VALOR"].isna().sum()
    if n_nulls > 0:
        print(f"  [WARN] {n_nulls} valores no convertibles en columna VALOR.")

    # --- Convertir columnas de fecha a datetime ---
    df["VIGENCIADESDE"] = pd.to_datetime(df["VIGENCIADESDE"])
    df["VIGENCIAHASTA"] = pd.to_datetime(df["VIGENCIAHASTA"])

    # --- Ordenar cronológicamente ---
    df = df.sort_values("VIGENCIADESDE").reset_index(drop=True)

    # --- Validación de rango ---
    min_date = df["VIGENCIADESDE"].min()
    max_date = df["VIGENCIAHASTA"].max()
    print(f"  [OK] Rango de datos: {min_date.strftime('%Y-%m-%d')} -> "
          f"{max_date.strftime('%Y-%m-%d')}")
    print(f"  [OK] {len(df):,} registros limpios | "
          f"Valor TCRM: {df['VALOR'].min():,.2f} - {df['VALOR'].max():,.2f} COP")

    return df


def build_daily_series(df: pd.DataFrame) -> pd.Series:
    """Construye la serie diaria completa de TCRM con imputación normativa.

    La Superintendencia Financiera de Colombia establece que la TCRM
    vigente para un día no transado (fin de semana o festivo) corresponde
    al último valor publicado. Esta función:

    1. Expande los registros con vigencia multi-día a filas individuales
       por cada día del intervalo ``[VIGENCIADESDE, VIGENCIAHASTA]``.
    2. Genera un índice diario continuo desde ``1991-12-02`` hasta
       ``2021-10-19`` (10,915 días calendario).
    3. Rellena los días sin TCRM publicada mediante **forward fill**.

    Args:
        df: DataFrame limpio de TCRM (salida de :func:`clean_tcrm`).

    Returns:
        :class:`~pandas.Series` con índice ``DatetimeIndex`` diario y
        valores ``float64``. Contiene exactamente 10,915 registros.

    Notes:
        Los 3,727 días no transados corresponden a fines de semana y
        festivos bancarios colombianos. Se imputan conforme a la
        Circular Reglamentaria DODM-146 de la Superintendencia Financiera.
    """
    print("[TCRM] Construyendo serie diaria con imputacion forward-fill...")

    # --- Expandir filas multi-día a días individuales ---
    records: list[dict] = []
    for _, row in df.iterrows():
        start = row["VIGENCIADESDE"]
        end = row["VIGENCIAHASTA"]
        valor = row["VALOR"]
        days_range = pd.date_range(start=start, end=end, freq="D")
        for d in days_range:
            records.append({"fecha": d, "tcrm": valor})

    df_expanded = pd.DataFrame(records)

    # Manejar posibles duplicados (en caso de solapamiento): mantener último
    df_expanded = df_expanded.drop_duplicates(subset="fecha", keep="last")

    # --- Crear índice diario completo ---
    full_idx = _generate_date_range(_START_DATE, _END_DATE, freq="D")

    # --- Reindexar y forward fill ---
    df_expanded = df_expanded.set_index("fecha")
    df_expanded = df_expanded.reindex(full_idx)
    df_expanded["tcrm"] = df_expanded["tcrm"].ffill()

    # --- Estadísticas ---
    n_traded = df_expanded["tcrm"].notna().sum()
    # Los NaN que puedan quedar serían al inicio si no hay dato previo
    n_imputed_initial = df_expanded["tcrm"].isna().sum()
    if n_imputed_initial > 0:
        # Backfill solo para el inicio si quedan NaN
        df_expanded["tcrm"] = df_expanded["tcrm"].bfill()
        print(f"  [INFO] {n_imputed_initial} dias al inicio imputados con backfill.")

    total = len(df_expanded)
    n_original = len(df)
    n_non_traded = total - n_original

    print(f"  [OK] Serie diaria generada: {total:,} dias calendario")
    print(f"       Registros originales     : {n_original:,}")
    print(f"       Dias no transados (ffill): {n_non_traded:,}")
    print(f"       Rango: {df_expanded.index.min().strftime('%Y-%m-%d')} -> "
          f"{df_expanded.index.max().strftime('%Y-%m-%d')}")

    tcrm_series = df_expanded["tcrm"].rename("TCRM")
    tcrm_series.index.name = "fecha"

    return tcrm_series


def export_tcrm_processed(series: pd.Series) -> Path:
    """Exporta la serie TCRM procesada a formato Parquet.

    Crea el directorio ``data/processed/`` si no existe.

    Args:
        series: Serie diaria de TCRM (salida de :func:`build_daily_series`).

    Returns:
        :class:`~pathlib.Path` del archivo exportado.
    """
    print("[TCRM] Exportando serie procesada...")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Convertir a DataFrame para preservar metadatos en Parquet
    df_out = series.to_frame(name="tcrm")
    df_out.index.name = "fecha"
    df_out.to_parquet(OUTPUT_FILE, index=True)

    size_kb = OUTPUT_FILE.stat().st_size / 1024
    print(f"  [OK] Exportado: {OUTPUT_FILE}")
    print(f"       Registros: {len(df_out):,} | Tamaño: {size_kb:.1f} KB")

    return OUTPUT_FILE


def get_tcrm_at_date(series: pd.Series, date_str: str) -> float | None:
    """Consulta el valor de la TCRM para una fecha específica.

    Args:
        series: Serie diaria de TCRM con ``DatetimeIndex``.
        date_str: Fecha en formato ``"YYYY-MM-DD"``.

    Returns:
        Valor TCRM como ``float``, o ``None`` si la fecha está fuera de rango.
    """
    target_date = pd.Timestamp(date_str)
    if target_date in series.index:
        return float(series.loc[target_date])
    print(f"  [WARN] Fecha {date_str} fuera del rango de la serie.")
    return None


def get_integrity_report(series: pd.Series, n_original: int) -> dict:
    """Genera un reporte de integridad de la serie TCRM.

    Args:
        series: Serie diaria de TCRM procesada.
        n_original: Número de registros en el dataset original.

    Returns:
        Diccionario con las claves ``total_dias_calendario``,
        ``registros_originales``, ``dias_no_transados``,
        ``fecha_inicio``, ``fecha_fin``.
    """
    return {
        "total_dias_calendario": len(series),
        "registros_originales": n_original,
        "dias_no_transados": len(series) - n_original,
        "fecha_inicio": series.index.min().strftime("%Y-%m-%d"),
        "fecha_fin": series.index.max().strftime("%Y-%m-%d"),
    }


# ===========================================================================
# Pipeline completo
# ===========================================================================

def run_tcrm_pipeline() -> pd.Series:
    """Orquesta el pipeline ETL completo de TCRM de extremo a extremo.

    Secuencia de ejecución:
    1. Carga del archivo Excel histórico.
    2. Limpieza y normalización de datos.
    3. Construcción de serie diaria con forward-fill normativo.
    4. Exportación a Parquet.

    Returns:
        Serie diaria de TCRM lista para análisis y forecasting.
    """
    print("=" * 60)
    print("  PIPELINE TCRM -- Analisis Tasa de Cambio Representativa")
    print("=" * 60)

    # 1. Carga
    df_raw = load_tcrm_raw()

    # 2. Limpieza
    df_clean = clean_tcrm(df_raw)
    n_original = len(df_clean)

    # 3. Serie diaria
    tcrm_series = build_daily_series(df_clean)

    # 4. Exportación
    export_tcrm_processed(tcrm_series)

    # --- Reporte final ---
    report = get_integrity_report(tcrm_series, n_original)
    print("=" * 60)
    print(f"  [TCRM] Pipeline completado exitosamente.")
    print(f"         Total dias calendario : {report['total_dias_calendario']:,}")
    print(f"         Registros originales  : {report['registros_originales']:,}")
    print(f"         Dias no transados     : {report['dias_no_transados']:,}")
    print(f"         TCRM inicial          : {tcrm_series.iloc[0]:,.2f} COP")
    print(f"         TCRM final            : {tcrm_series.iloc[-1]:,.2f} COP")
    print("=" * 60)

    return tcrm_series


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_tcrm_pipeline()
