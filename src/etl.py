"""
ETL Pipeline — prueba-tecnica-pragma
=====================================
Módulo para ingestión, limpieza, enriquecimiento y exportación de datos
de comercios y transacciones bancarias API Pragma.

Genera métricas RFM a nivel comercio y exporta el dataset consolidado
listo para segmentación (K-Means / clustering).

Uso:
    python src/etl.py
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Rutas base — sin hardcoding de rutas absolutas
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

COMERCIOS_FILE = RAW_DIR / "DataTransacciones_PruebasIngreso_Comercios.csv"
TRANSACCIONES_FILE = RAW_DIR / "DataTransacciones_PruebasIngreso_Transacciones.csv"

# ---------------------------------------------------------------------------
# Diccionario de meses en español (abreviados)
# ---------------------------------------------------------------------------
_ES_MONTHS: dict[str, int] = {
    "ene": 1, "feb": 2, "mar": 3, "abr": 4,
    "may": 5, "jun": 6, "jul": 7, "ago": 8,
    "sep": 9, "oct": 10, "nov": 11, "dic": 12,
}

# Comisión Pragma API sobre el monto transaccionado
_COMMISSION_RATE: float = 0.05


# ===========================================================================
# Helpers privados
# ===========================================================================

def _parse_spanish_date(date_str: str) -> datetime | None:
    """Convierte una fecha en español abreviado al tipo datetime de Python.

    Args:
        date_str: Cadena con formato ``"mar. 26- 2021 12:40pm"``.

    Returns:
        Objeto :class:`datetime` o ``None`` si el formato no coincide.

    Example:
        >>> _parse_spanish_date("mar. 26- 2021 12:40pm")
        datetime.datetime(2021, 3, 26, 12, 40)
    """
    if pd.isna(date_str) or str(date_str).strip() == "":
        return None

    pattern = r"(\w+)\.\s+(\d+)-\s*(\d{4})\s+(\d+):(\d+)(am|pm)"
    match = re.match(pattern, str(date_str).strip(), re.IGNORECASE)
    if not match:
        return None

    month_abbr, day, year, hour, minute, meridiem = match.groups()
    month = _ES_MONTHS.get(month_abbr.lower())
    if not month:
        return None

    hour_int = int(hour)
    if meridiem.lower() == "pm" and hour_int != 12:
        hour_int += 12
    elif meridiem.lower() == "am" and hour_int == 12:
        hour_int = 0

    return datetime(int(year), month, int(day), hour_int, int(minute))


def _clean_amount(value) -> float:
    """Convierte un monto colombiano formateado a float.

    Elimina el separador de miles (punto) utilizado en Colombia.

    Args:
        value: Valor original, por ejemplo ``"6.542.000"`` o ``6542000``.

    Returns:
        Monto como :class:`float`, o ``numpy.nan`` si el valor es nulo.

    Example:
        >>> _clean_amount("6.542.000")
        6542000.0
    """
    if pd.isna(value):
        return np.nan
    cleaned = str(value).strip().replace(".", "").replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza los nombres de columnas a snake_case limpio.

    Convierte a minúsculas, reemplaza espacios y caracteres especiales
    por guiones bajos, y elimina guiones bajos dobles o al final.

    Args:
        df: DataFrame con nombres de columnas originales.

    Returns:
        DataFrame con columnas renombradas en snake_case.
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
        # Colapsar múltiples guiones bajos consecutivos
        normalized = re.sub(r"_+", "_", normalized).strip("_")
        new_cols.append(normalized)
    df.columns = new_cols
    return df


# ===========================================================================
# Funciones principales del pipeline
# ===========================================================================

def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga los archivos CSV crudos de comercios y transacciones.

    Lee ambos archivos desde :data:`RAW_DIR`, normaliza los nombres de
    columnas a snake_case y realiza validaciones básicas de duplicados
    y nulos en la clave primaria.

    Returns:
        Tupla ``(df_comercios, df_transacciones)`` con columnas normalizadas.

    Raises:
        FileNotFoundError: Si alguno de los archivos no existe en ``RAW_DIR``.
    """
    print("[ETL] Cargando datos crudos...")

    try:
        df_comercios = pd.read_csv(
            COMERCIOS_FILE,
            sep=";",
            encoding="latin-1",
            dtype=str,  # Cargar todo como string para limpieza controlada
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"Archivo no encontrado: {COMERCIOS_FILE}")

    try:
        df_transacciones = pd.read_csv(
            TRANSACCIONES_FILE,
            sep=";",
            encoding="latin-1",
            dtype=str,
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"Archivo no encontrado: {TRANSACCIONES_FILE}")

    df_comercios = _normalize_column_names(df_comercios)
    df_transacciones = _normalize_column_names(df_transacciones)

    # --- Validación de duplicados en clave primaria ---
    dup_comercios = df_comercios["id_comercio"].duplicated().sum()
    if dup_comercios > 0:
        print(f"  [WARN] {dup_comercios} filas duplicadas en id_comercio (comercios).")

    dup_trans = df_transacciones.duplicated(
        subset=["id_comercio", "mes_anio_transaccion"]
    ).sum()
    if dup_trans > 0:
        print(f"  [WARN] {dup_trans} combinaciones duplicadas en transacciones.")

    # --- Validación de nulos en claves ---
    null_ids_c = df_comercios["id_comercio"].isna().sum()
    null_ids_t = df_transacciones["id_comercio"].isna().sum()
    if null_ids_c > 0:
        print(f"  [WARN] {null_ids_c} filas sin id_comercio en comercios.")
    if null_ids_t > 0:
        print(f"  [WARN] {null_ids_t} filas sin id_comercio en transacciones.")

    print(f"  [OK] Comercios: {len(df_comercios):,} filas | "
          f"Transacciones: {len(df_transacciones):,} filas")

    return df_comercios, df_transacciones


def clean_comercios(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia y normaliza el DataFrame de comercios.

    Operaciones aplicadas:
    - Parseo de columnas de fecha en formato español a ``datetime``.
    - Conversión de montos colombianos (``"6.542.000"``) a ``float``.
    - Normalización de ciudades: strip de espacios múltiples.
    - Eliminación de duplicados por ``id_comercio`` (mantiene primer registro).
    - Selección de columnas relevantes para el análisis.

    Args:
        df: DataFrame crudo de comercios con columnas en snake_case.

    Returns:
        DataFrame limpio y tipado listo para enriquecimiento RFM.
    """
    print("[ETL] Limpiando comercios...")

    df = df.copy()

    # --- Parseo de fechas en español ---
    date_cols = {
        "fecha_vinculacion_datafono": "fecha_vinculacion",
        "fecha_de_creacion_comercio": "fecha_creacion",
        "fecha_1ra_transaccion": "fecha_primera_transaccion",
    }
    for raw_col, clean_col in date_cols.items():
        if raw_col in df.columns:
            df[clean_col] = df[raw_col].apply(_parse_spanish_date)
            df[clean_col] = pd.to_datetime(df[clean_col])

    # --- Conversión de montos ---
    amount_cols = {
        "monto_ventas_acumulado": "monto_ventas_acumulado",
        "monto_ventas_ultimo_mes": "monto_ventas_ultimo_mes",
    }
    for raw_col, clean_col in amount_cols.items():
        if raw_col in df.columns:
            df[clean_col] = df[raw_col].apply(_clean_amount)

    # --- Normalización de texto: ciudades y categorías ---
    text_cols = ["ciudad", "categoria_comercio", "sub_categoria_comercio"]
    for col in text_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)  # Colapsar espacios múltiples
            )

    # --- Eliminar duplicados por id_comercio ---
    before = len(df)
    df = df.drop_duplicates(subset="id_comercio", keep="first").reset_index(drop=True)
    removed = before - len(df)
    if removed > 0:
        print(f"  [INFO] {removed} duplicados eliminados en comercios.")

    # --- Selección de columnas relevantes ---
    keep_cols = [
        "id_comercio",
        "id_terminal",
        "categoria_comercio",
        "sub_categoria_comercio",
        "ciudad",
        "fecha_vinculacion",
        "fecha_creacion",
        "fecha_primera_transaccion",
        "monto_ventas_acumulado",
        "monto_ventas_ultimo_mes",
    ]
    # Filtrar solo las columnas que efectivamente existen
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    print(f"  [OK] Comercios limpios: {len(df):,} registros | "
          f"{len(df.columns)} columnas retenidas")

    return df


def clean_transacciones(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia y filtra el DataFrame de transacciones.

    Operaciones aplicadas:
    - Conversión de ``"mes-anio"`` (ej. ``"3-2021"``) a ``datetime``
      representando el primer día del mes.
    - Conversión de ``cantidad_transacciones`` a entero.
    - Filtrado de los últimos 6 meses tomando la fecha máxima del
      dataset como punto de corte.

    Args:
        df: DataFrame crudo de transacciones con columnas en snake_case.

    Returns:
        DataFrame filtrado con columna ``fecha_transaccion`` como datetime.
    """
    print("[ETL] Limpiando transacciones...")

    df = df.copy()

    # --- Convertir "mes-anio" al primer día del mes ---
    def _parse_mes_anio(val: str) -> pd.Timestamp | None:
        """Parsea 'mes-anio' (ej. '3-2021') a Timestamp del primer día del mes."""
        try:
            parts = str(val).strip().split("-")
            month, year = int(parts[0]), int(parts[1])
            return pd.Timestamp(year=year, month=month, day=1)
        except (ValueError, IndexError):
            return pd.NaT

    df["fecha_transaccion"] = df["mes_anio_transaccion"].apply(_parse_mes_anio)

    # --- Convertir cantidad a numérico ---
    df["cantidad_transacciones"] = pd.to_numeric(
        df["cantidad_transacciones"], errors="coerce"
    ).fillna(0).astype(int)

    # --- Filtrar últimos 6 meses desde la fecha máxima del dataset ---
    max_date = df["fecha_transaccion"].max()
    cutoff_date = max_date - pd.DateOffset(months=6) + pd.DateOffset(days=1)

    before = len(df)
    df = df[df["fecha_transaccion"] >= cutoff_date].reset_index(drop=True)
    filtered_out = before - len(df)

    print(f"  [INFO] Fecha máxima del dataset: {max_date.strftime('%Y-%m-%d')}")
    print(f"  [INFO] Corte 6 meses desde: {cutoff_date.strftime('%Y-%m-%d')}")
    if filtered_out > 0:
        print(f"  [INFO] {filtered_out:,} filas excluidas fuera del período.")

    print(f"  [OK] Transacciones limpias: {len(df):,} filas | "
          f"{df['id_comercio'].nunique():,} comercios únicos")

    return df


def generate_rfm(
    df_comercios: pd.DataFrame,
    df_transacciones: pd.DataFrame,
) -> pd.DataFrame:
    """Calcula métricas RFM y KPIs bancarios a nivel de comercio.

    Métricas generadas:

    - **Recencia**: Días transcurridos desde la última transacción del
      comercio hasta la fecha máxima del dataset. Valores bajos indican
      mayor actividad reciente.
    - **Frecuencia**: Número total de transacciones en los últimos 6 meses.
    - **Monetary**: Monto acumulado de ventas del comercio (proxy monetario,
      ya que el dataset de transacciones no incluye montos por registro).
    - **Ticket Promedio**: ``Monetary / Frecuencia``.
    - **Comisión Generada**: ``Monetary * 0.05`` (tasa de comisión API Pragma).
    - **Promedio Mensual**: ``Frecuencia / meses_activos``.

    Args:
        df_comercios: DataFrame limpio de comercios (debe contener
            ``id_comercio`` y ``monto_ventas_acumulado``).
        df_transacciones: DataFrame limpio de transacciones con columnas
            ``id_comercio``, ``fecha_transaccion`` y ``cantidad_transacciones``.

    Returns:
        DataFrame con una fila por comercio y las columnas RFM calculadas.
    """
    print("[ETL] Calculando métricas RFM...")

    # --- Fecha de referencia: máximo del dataset de transacciones ---
    max_date = df_transacciones["fecha_transaccion"].max()
    print(f"  [INFO] Fecha de referencia RFM: {max_date.strftime('%Y-%m-%d')}")

    # --- Recencia: días desde la última transacción hasta max_date ---
    recencia = (
        df_transacciones
        .groupby("id_comercio")["fecha_transaccion"]
        .max()
        .rename("ultima_transaccion")
        .reset_index()
    )
    recencia["recencia_dias"] = (max_date - recencia["ultima_transaccion"]).dt.days

    # --- Frecuencia: suma de transacciones en los últimos 6 meses ---
    frecuencia = (
        df_transacciones
        .groupby("id_comercio")["cantidad_transacciones"]
        .sum()
        .rename("frecuencia")
        .reset_index()
    )

    # --- Meses activos: número de períodos distintos por comercio ---
    meses_activos = (
        df_transacciones
        .groupby("id_comercio")["fecha_transaccion"]
        .nunique()
        .rename("meses_activos")
        .reset_index()
    )

    # --- Monetary: monto acumulado de ventas (fuente: comercios) ---
    monetary = df_comercios[["id_comercio", "monto_ventas_acumulado"]].copy()
    monetary = monetary.rename(columns={"monto_ventas_acumulado": "monetary"})

    # --- Unir todas las métricas RFM ---
    df_rfm = (
        recencia
        .merge(frecuencia, on="id_comercio", how="left")
        .merge(meses_activos, on="id_comercio", how="left")
        .merge(monetary, on="id_comercio", how="left")
    )

    # --- Métricas derivadas ---
    df_rfm["ticket_promedio"] = np.where(
        df_rfm["frecuencia"] > 0,
        df_rfm["monetary"] / df_rfm["frecuencia"],
        np.nan,
    )
    df_rfm["comision_generada"] = df_rfm["monetary"] * _COMMISSION_RATE

    df_rfm["promedio_transacciones_mensual"] = np.where(
        df_rfm["meses_activos"] > 0,
        df_rfm["frecuencia"] / df_rfm["meses_activos"],
        np.nan,
    )

    # --- Redondear métricas monetarias ---
    for col in ["monetary", "ticket_promedio", "comision_generada",
                "promedio_transacciones_mensual"]:
        if col in df_rfm.columns:
            df_rfm[col] = df_rfm[col].round(2)

    print(f"  [OK] RFM calculado para {len(df_rfm):,} comercios")
    print(f"       Recencia promedio  : {df_rfm['recencia_dias'].mean():.1f} días")
    print(f"       Frecuencia promedio: {df_rfm['frecuencia'].mean():.1f} transacciones")
    print(f"       Monetary promedio  : ${df_rfm['monetary'].mean():,.0f}")

    return df_rfm


def merge_datasets(
    df_comercios: pd.DataFrame,
    df_rfm: pd.DataFrame,
) -> pd.DataFrame:
    """Une el dataset de comercios con las métricas RFM calculadas.

    Realiza un ``left join`` desde comercios hacia RFM para conservar
    todos los comercios, incluso aquellos sin transacciones registradas.
    Valida la consistencia de IDs antes y después del merge.

    Args:
        df_comercios: DataFrame limpio de comercios.
        df_rfm: DataFrame con métricas RFM por ``id_comercio``.

    Returns:
        DataFrame consolidado con información de comercios + RFM,
        listo para ser segmentado.
    """
    print("[ETL] Consolidando datasets...")

    ids_comercios = set(df_comercios["id_comercio"].unique())
    ids_rfm = set(df_rfm["id_comercio"].unique())

    ids_sin_rfm = ids_comercios - ids_rfm
    ids_rfm_sin_comercio = ids_rfm - ids_comercios

    if ids_sin_rfm:
        print(f"  [WARN] {len(ids_sin_rfm):,} comercios sin transacciones en el período.")
    if ids_rfm_sin_comercio:
        print(f"  [WARN] {len(ids_rfm_sin_comercio):,} IDs en RFM sin registro en comercios.")

    # Left join: comercios como tabla base
    df_merged = df_comercios.merge(df_rfm, on="id_comercio", how="left")

    # Rellenar métricas nulas con cero para comercios sin transacciones
    fill_cols = ["frecuencia", "recencia_dias", "meses_activos"]
    for col in fill_cols:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].fillna(0)

    print(f"  [OK] Dataset consolidado: {len(df_merged):,} filas | "
          f"{len(df_merged.columns)} columnas")

    return df_merged


def export_processed(df: pd.DataFrame) -> Path:
    """Exporta el dataset final procesado a ``/data/processed/``.

    Crea el directorio de destino si no existe. Exporta sin índice
    y con encoding UTF-8 para compatibilidad con herramientas BI.

    Args:
        df: DataFrame consolidado listo para exportar.

    Returns:
        :class:`~pathlib.Path` del archivo exportado.
    """
    print("[ETL] Exportando dataset procesado...")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_DIR / "comercios_segmented_prep.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")

    size_kb = output_path.stat().st_size / 1024
    print(f"  [OK] Exportado: {output_path}")
    print(f"       Filas: {len(df):,} | Columnas: {len(df.columns)} | "
          f"Tamaño: {size_kb:.1f} KB")

    return output_path


def run_etl() -> pd.DataFrame:
    """Orquesta el pipeline ETL completo de extremo a extremo.

    Secuencia de ejecución:
    1. Carga de datos crudos.
    2. Limpieza de comercios y transacciones.
    3. Cálculo de métricas RFM.
    4. Consolidación de datasets.
    5. Exportación del archivo procesado.

    Returns:
        DataFrame final exportado (para uso interactivo o en notebooks).
    """
    print("=" * 60)
    print("  PIPELINE ETL — API Pragma / Análisis de Comercios")
    print("=" * 60)

    # 1. Ingesta
    df_comercios_raw, df_transacciones_raw = load_raw_data()

    # 2. Limpieza
    df_comercios = clean_comercios(df_comercios_raw)
    df_transacciones = clean_transacciones(df_transacciones_raw)

    # 3. RFM
    df_rfm = generate_rfm(df_comercios, df_transacciones)

    # 4. Consolidación
    df_final = merge_datasets(df_comercios, df_rfm)

    # 5. Exportación
    export_processed(df_final)

    print("=" * 60)
    print("  [ETL] Pipeline completado exitosamente.")
    print("=" * 60)

    return df_final


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_etl()
