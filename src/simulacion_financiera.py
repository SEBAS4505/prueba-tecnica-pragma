"""
Simulacion Financiera — Puente Financiero / Waterfall Integrado
================================================================

Genera una tabla granular a nivel de palanca/componente donde, para CADA
escenario (Conservador, Intermedio, Acelerado), la suma incremental de las
comisiones (Delta COP) explica exactamente el 100% de la meta de crecimiento
(+50% de comision actual = ~COP $330,766,258.30).

Estructura:
  Pilar 1 — Cartera Actual (Retencion y Crecimiento):
    - VIP - Blindaje (Fase 1: Jun-Ago)
    - En Riesgo - Retencion (Fase 1: Jun-Ago)
    - Core - Expansion (Fase 2: Sep-Nov)
    - Ocasional - Reactivacion/Ticket Alto (Fase 2: Sep-Nov)

  Pilar 2 — Adquisicion Neta (Captacion Look-Alike):
    - Captacion Neta Look-Alike VIP (Fase 1 y 2: Jun-Dic)
    - Captacion Neta Look-Alike Core (Fase 1 y 2: Jun-Dic)

Output: data/processed/simulacion_puente_powerbi.csv
        (1 fila por combinacion escenario x componente)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. Configuracion de rutas
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
INPUT_CSV = DATA_DIR / "comercios_final_k_medoids.csv"
OUTPUT_CSV = DATA_DIR / "simulacion_puente_powerbi.csv"

# ---------------------------------------------------------------------------
# 1. Carga de datos y baseline global
# ---------------------------------------------------------------------------
df = pd.read_csv(INPUT_CSV)
print(f"[OK] Dataset cargado: {len(df):,} comercios x {df.shape[1]} columnas")

SEGMENT_ORDER = [
    "Champions / VIP",
    "Core / Crecimiento",
    "En Riesgo / Churn",
    "Ocasional / Ticket Alto",
]

COMISION_ACTUAL_GLOBAL = df["commission"].sum()
META_MULTIPLICADOR = 1.50
COMISION_META_GLOBAL = COMISION_ACTUAL_GLOBAL * META_MULTIPLICADOR
GAP_ABSOLUTO_GLOBAL = COMISION_META_GLOBAL - COMISION_ACTUAL_GLOBAL

print(f"   Comision actual global : COP $ {COMISION_ACTUAL_GLOBAL:>15,.2f}")
print(f"   Meta +50%              : COP $ {COMISION_META_GLOBAL:>15,.2f}")
print(f"   Gap absoluto           : COP $ {GAP_ABSOLUTO_GLOBAL:>15,.2f}")

# ---------------------------------------------------------------------------
# 2. Metricas consolidadas por segmento
# ---------------------------------------------------------------------------
seg_stats = (
    df.groupby("segmento")
      .agg(
          n_comercios=("id_comercio", "count"),
          comision_actual_segmento=("commission", "sum"),
      )
      .reindex(SEGMENT_ORDER)
      .reset_index()
)

seg_stats["comision_promedio_comercio"] = (
    seg_stats["comision_actual_segmento"] / seg_stats["n_comercios"]
)

# Índice rápido para lookup
_seg_dict = {
    row["segmento"]: row
    for _, row in seg_stats.iterrows()
}

print("\n[OK] Metricas consolidadas por segmento:")
for _, r in seg_stats.iterrows():
    print(f"   {r['segmento']:<25s}  "
          f"n={r['n_comercios']:>5,}  "
          f"comision=COP ${r['comision_actual_segmento']:>15,.2f}  "
          f"prom/comercio=COP ${r['comision_promedio_comercio']:>10,.2f}")

# ---------------------------------------------------------------------------
# 3. Definicion de escenarios con tasas de uplift/retencion
# ---------------------------------------------------------------------------
SCENARIOS = {
    "Conservador": {
        "Champions / VIP":          0.10,
        "Core / Crecimiento":       0.10,
        "En Riesgo / Churn":        0.05,
        "Ocasional / Ticket Alto":  0.00,
    },
    "Intermedio": {
        "Champions / VIP":          0.15,
        "Core / Crecimiento":       0.20,
        "En Riesgo / Churn":        0.10,
        "Ocasional / Ticket Alto":  0.05,
    },
    "Acelerado": {
        "Champions / VIP":          0.25,
        "Core / Crecimiento":       0.30,
        "En Riesgo / Churn":        0.20,
        "Ocasional / Ticket Alto":  0.10,
    },
}

# ---------------------------------------------------------------------------
# 4. Definicion de componentes (palancas) del Waterfall
# ---------------------------------------------------------------------------
# Pilar 1: Componentes de cartera actual
COMPONENTES_CARTERA = [
    {
        "segmento":        "Champions / VIP",
        "componente":      "VIP - Blindaje",
        "fase_ejecucion":  "Fase 1: Blindaje (Jun-Ago 2021)",
        "kpi_template":    "+{tasa_pct:.0f}% uplift en comision / GMV",
    },
    {
        "segmento":        "En Riesgo / Churn",
        "componente":      "En Riesgo - Retencion",
        "fase_ejecucion":  "Fase 1: Retencion (Jun-Ago 2021)",
        "kpi_template":    "Retener {tasa_pct:.0f}% de comision en riesgo",
    },
    {
        "segmento":        "Core / Crecimiento",
        "componente":      "Core - Expansion",
        "fase_ejecucion":  "Fase 2: Expansion (Sep-Nov 2021)",
        "kpi_template":    "+{tasa_pct:.0f}% uplift en comision / GMV",
    },
    {
        "segmento":        "Ocasional / Ticket Alto",
        "componente":      "Ocasional - Reactivacion / Ticket Alto",
        "fase_ejecucion":  "Fase 2: Reactivacion (Sep-Nov 2021)",
        "kpi_template":    "+{tasa_pct:.0f}% uplift en comision / GMV",
    },
]

# Pilar 2: Componentes de adquisicion neta
COMPONENTES_CAPTACION = [
    {
        "componente":      "Captacion Neta Look-Alike VIP",
        "fase_ejecucion":  "Fase 1 y 2: Captacion (Jun-Dic 2021)",
        "segmento_ref":    "Champions / VIP",
        "split_pct":       0.50,
    },
    {
        "componente":      "Captacion Neta Look-Alike Core",
        "fase_ejecucion":  "Fase 1 y 2: Captacion (Jun-Dic 2021)",
        "segmento_ref":    "Core / Crecimiento",
        "split_pct":       0.50,
    },
]

MESES_CAPTACION = 7  # Jun-Dic = 7 meses

# ---------------------------------------------------------------------------
# 5. Construccion de la tabla Waterfall
# ---------------------------------------------------------------------------
filas = []

for escenario, tasas in SCENARIOS.items():

    # --- Pilar 1: Cartera Actual ---
    delta_cartera_total = 0.0
    for comp in COMPONENTES_CARTERA:
        seg = comp["segmento"]
        tasa = tasas[seg]
        seg_data = _seg_dict[seg]

        comision_base = seg_data["comision_actual_segmento"]
        delta = comision_base * tasa
        delta_cartera_total += delta

        tasa_pct = tasa * 100
        kpi = comp["kpi_template"].format(tasa_pct=tasa_pct)

        filas.append({
            "escenario":                     escenario,
            "pilar_estrategico":             "1. Cartera Actual",
            "componente":                    comp["componente"],
            "fase_ejecucion":                comp["fase_ejecucion"],
            "comercios_objetivo":            int(seg_data["n_comercios"]),
            "kpi_meta_operativa":            kpi,
            "comision_base_cop":             comision_base,
            "delta_incremental_cop":         delta,
            "pct_contribucion_gap":          delta / GAP_ABSOLUTO_GLOBAL,
            "comision_proyectada_total_cop": comision_base + delta,
            "gap_absoluto_global_cop":       GAP_ABSOLUTO_GLOBAL,
        })

    # --- Pilar 2: Adquisicion Neta ---
    gap_remanente = GAP_ABSOLUTO_GLOBAL - delta_cartera_total
    # El gap remanente se asigna a los componentes de captacion segun split_pct
    for comp in COMPONENTES_CAPTACION:
        seg_ref = comp["segmento_ref"]
        seg_data = _seg_dict[seg_ref]
        comision_prom = seg_data["comision_promedio_comercio"]

        delta_captacion = gap_remanente * comp["split_pct"]
        nuevos_comercios = delta_captacion / comision_prom
        nuevos_mensuales = nuevos_comercios / MESES_CAPTACION

        kpi = (
            f"Vincular {nuevos_comercios:.0f} comercios "
            f"Look-Alike {seg_ref.split(' / ')[0]} "
            f"(~{nuevos_mensuales:.0f}/mes)"
        )

        filas.append({
            "escenario":                     escenario,
            "pilar_estrategico":             "2. Adquisicion Neta",
            "componente":                    comp["componente"],
            "fase_ejecucion":                comp["fase_ejecucion"],
            "comercios_objetivo":            int(round(nuevos_comercios)),
            "kpi_meta_operativa":            kpi,
            "comision_base_cop":             0.0,
            "delta_incremental_cop":         delta_captacion,
            "pct_contribucion_gap":          delta_captacion / GAP_ABSOLUTO_GLOBAL,
            "comision_proyectada_total_cop": delta_captacion,
            "gap_absoluto_global_cop":       GAP_ABSOLUTO_GLOBAL,
        })

df_out = pd.DataFrame(filas)

# ---------------------------------------------------------------------------
# 6. Redondeo y formato
# ---------------------------------------------------------------------------
COLS_MONETARIAS = [
    "comision_base_cop",
    "delta_incremental_cop",
    "comision_proyectada_total_cop",
    "gap_absoluto_global_cop",
]
COLS_PORCENTAJE = ["pct_contribucion_gap"]

for c in COLS_MONETARIAS:
    df_out[c] = df_out[c].round(2)
for c in COLS_PORCENTAJE:
    df_out[c] = df_out[c].round(4)
# comercios_objetivo ya es int; asegurar tipo
df_out["comercios_objetivo"] = df_out["comercios_objetivo"].astype("Int64")

# ---------------------------------------------------------------------------
# 7. Validaciones con assert
# ---------------------------------------------------------------------------
# 7a. Sin valores nulos
nulos = df_out.isnull().sum()
assert not nulos.any(), (
    f"VALIDATION FAILED: Existen valores nulos en columnas:\n{nulos[nulos > 0]}"
)
print("\n[OK] Sin valores nulos en el dataset de salida")

# 7b. Suma de delta_incremental_cop == GAP_ABSOLUTO_GLOBAL para cada escenario
#     (tolerancia < $1.0 COP)
print("\n[VALIDACION] Suma de deltas incrementales por escenario:")
all_ok = True
for escenario in SCENARIOS:
    mask = df_out["escenario"] == escenario
    delta_sum = df_out.loc[mask, "delta_incremental_cop"].sum()
    diff = abs(delta_sum - GAP_ABSOLUTO_GLOBAL)
    status = "OK" if diff < 1.0 else "FAIL"
    if diff >= 1.0:
        all_ok = False
    print(f"   {escenario:<15s}: sum(delta) = COP $ {delta_sum:>15,.2f}  "
          f"(gap = {GAP_ABSOLUTO_GLOBAL:,.2f}, diff = {diff:,.2f})  [{status}]")

assert all_ok, "VALIDATION FAILED: La suma de delta_incremental_cop no cierra al 100% del gap en al menos un escenario."

# 7c. Verificar que pct_contribucion_gap suma ~1.0 por escenario
print("\n[VALIDACION] Suma de pct_contribucion_gap por escenario:")
for escenario in SCENARIOS:
    mask = df_out["escenario"] == escenario
    pct_sum = df_out.loc[mask, "pct_contribucion_gap"].sum()
    print(f"   {escenario:<15s}: sum(pct) = {pct_sum:.4f}  "
          f"(diff de 1.0 = {abs(pct_sum - 1.0):.6f})  "
          f"[{'OK' if abs(pct_sum - 1.0) < 1e-3 else 'WARN'}]")

# ---------------------------------------------------------------------------
# 8. Exportacion
# ---------------------------------------------------------------------------
# Reordenar columnas segun la especificacion exacta
COL_ORDER = [
    "escenario",
    "pilar_estrategico",
    "componente",
    "fase_ejecucion",
    "comercios_objetivo",
    "kpi_meta_operativa",
    "comision_base_cop",
    "delta_incremental_cop",
    "pct_contribucion_gap",
    "comision_proyectada_total_cop",
    "gap_absoluto_global_cop",
]
df_out = df_out[COL_ORDER]

df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
print(f"\n[OK] Archivo exportado: {OUTPUT_CSV}")
print(f"   {len(df_out)} filas x {df_out.shape[1]} columnas")

# ---------------------------------------------------------------------------
# 9. Resumen final
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("RESUMEN — PUENTE FINANCIERO / WATERFALL POWER BI")
print("=" * 70)
print(f"   Comision actual      : COP $ {COMISION_ACTUAL_GLOBAL:>15,.2f}")
print(f"   Meta +50%            : COP $ {COMISION_META_GLOBAL:>15,.2f}")
print(f"   Gap a cerrar         : COP $ {GAP_ABSOLUTO_GLOBAL:>15,.2f}")
print()

for escenario in SCENARIOS:
    mask = df_out["escenario"] == escenario
    mask_cartera = mask & (df_out["pilar_estrategico"] == "1. Cartera Actual")
    mask_captacion = mask & (df_out["pilar_estrategico"] == "2. Adquisicion Neta")

    delta_cart = df_out.loc[mask_cartera, "delta_incremental_cop"].sum()
    delta_capt = df_out.loc[mask_captacion, "delta_incremental_cop"].sum()

    print(f"  --- {escenario} ---")
    print(f"     Cartera Actual    : COP $ {delta_cart:>15,.2f} "
          f"({delta_cart / GAP_ABSOLUTO_GLOBAL * 100:.1f}%)")
    print(f"     Adquisicion Neta  : COP $ {delta_capt:>15,.2f} "
          f"({delta_capt / GAP_ABSOLUTO_GLOBAL * 100:.1f}%)")
    print(f"     Total             : COP $ {delta_cart + delta_capt:>15,.2f} "
          f"({(delta_cart + delta_capt) / GAP_ABSOLUTO_GLOBAL * 100:.1f}%)")

    # Mostrar KPI de comercios nuevos por escenario
    for comp in COMPONENTES_CAPTACION:
        row = df_out[(df_out["escenario"] == escenario)
                     & (df_out["componente"] == comp["componente"])].iloc[0]
        print(f"       -> {comp['componente']}: "
              f"{row['comercios_objetivo']} comercios nuevos "
              f"(COP $ {row['delta_incremental_cop']:,.2f})")
    print()
