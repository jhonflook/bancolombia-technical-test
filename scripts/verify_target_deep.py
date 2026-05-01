"""
Análisis profundo de discordancias en var_rta.
Foco: por qué 16,032 clase-1 tienen total_pago_3m=0 y qué explica la clasificación.
"""
import pandas as pd
import numpy as np

CLIENTES  = "datalake/clientes_seleccionados_prueba.csv"
PAGOS     = "datalake/debitos_dev_cruce_tanque_poblacion_prueba_tecnica.csv"
CANALES   = "datalake/debitos_dev_cruce_canales_poblacion_sel_vars_prueba_tecnica_1.csv"
MORAS     = "datalake/debitos_dev_cruce_moras_poblacion_prueba_tecnica.csv"
JOIN_KEYS = ["num_doc", "obl17", "f_analisis"]

print("Cargando datos...")
clientes = pd.read_csv(CLIENTES, dtype={k: str for k in JOIN_KEYS})
pagos    = pd.read_csv(PAGOS,    dtype={k: str for k in JOIN_KEYS}).drop_duplicates()
moras    = pd.read_csv(MORAS,    dtype={k: str for k in JOIN_KEYS}).drop_duplicates()

df = clientes.merge(pagos, on=JOIN_KEYS, how="left") \
             .merge(moras, on=JOIN_KEYS, how="left")

# Totales por canal y ventana
for w in ["3m", "6m", "9m", "12m"]:
    d = df.get(f"avg_pago_debito_{w}", 0).fillna(0)
    f_= df.get(f"avg_pago_fisico_{w}", 0).fillna(0)
    v = df.get(f"avg_pago_virtual_{w}", 0).fillna(0)
    o = df.get(f"avg_pago_otros_{w}", 0).fillna(0)
    df[f"total_pago_{w}"] = d + f_ + v + o
    df[f"prop_debito_{w}"] = np.where(df[f"total_pago_{w}"] > 0,
                                       d / df[f"total_pago_{w}"], 0.0)

# ── ¿La clase 0 siempre tiene total_pago=0? ──────────────────────────────────
print("\n── ¿Clase 0 tiene algún pago en alguna ventana? ──────────────────────────")
clase0 = df[df.var_rta == 0]
for w in ["3m", "6m", "9m", "12m"]:
    n_con_pago = (clase0[f"total_pago_{w}"] > 0).sum()
    print(f"  Ventana {w}: {n_con_pago:,} filas clase-0 con total_pago > 0  "
          f"({n_con_pago/len(clase0):.2%})")

# ── Clase 0 con pagos en alguna ventana: ¿en qué canal? ───────────────────────
print("\n── Clase 0 CON pagos en 12m — distribución por canal ──────────────────────")
c0_con_pago = clase0[clase0["total_pago_12m"] > 0]
print(f"  Filas: {len(c0_con_pago):,}")
if len(c0_con_pago) > 0:
    for col in ["avg_pago_debito_12m","avg_pago_fisico_12m",
                "avg_pago_virtual_12m","avg_pago_otros_12m"]:
        n = (c0_con_pago[col] > 0).sum()
        print(f"    {col}: {n:,} filas > 0  ({n/len(c0_con_pago):.2%})")

# ── Clase 1 sin ningún pago en ninguna ventana ───────────────────────────────
clase1 = df[df.var_rta == 1]
c1_sin_pago = clase1[clase1["total_pago_12m"] == 0]
print(f"\n── Clase 1 sin pago en ventana 12m: {len(c1_sin_pago):,} filas "
      f"({len(c1_sin_pago)/len(clase1):.2%})")

# ── Verificar si recurrencia se calcula sobre los 17 periodos por obligación ──
print("\n── Reconstrucción de recurrencia agregada por obligación (cross-periodo) ──")
# Para cada obligación única, contar en cuántos periodos tiene pago_debito > 0
pagos_unique = pagos.copy()
pagos_unique["tiene_debito_3m"] = (pagos_unique["avg_pago_debito_3m"].fillna(0) > 0).astype(int)
pagos_unique["tiene_pago_3m"]   = ((pagos_unique["avg_pago_debito_3m"].fillna(0) +
                                    pagos_unique["avg_pago_fisico_3m"].fillna(0) +
                                    pagos_unique["avg_pago_virtual_3m"].fillna(0) +
                                    pagos_unique["avg_pago_otros_3m"].fillna(0)) > 0).astype(int)

# Agrupar por obligación (colapsar todos los periodos)
obl_stats = pagos_unique.groupby(["num_doc","obl17"]).agg(
    periodos_total=("f_analisis", "count"),
    periodos_con_debito=("tiene_debito_3m", "sum"),
    periodos_con_pago=("tiene_pago_3m", "sum"),
).reset_index()
obl_stats["recurrencia_debito"] = np.where(
    obl_stats["periodos_con_pago"] > 0,
    obl_stats["periodos_con_debito"] / obl_stats["periodos_con_pago"],
    0.0
)
obl_stats["recurrencia_sobre_total"] = (obl_stats["periodos_con_debito"] /
                                         obl_stats["periodos_total"])

# Unir a clientes (colapsar periodos -> una fila por obligación)
clientes_obl = clientes.groupby(["num_doc","obl17"])["var_rta"].max().reset_index()
obl_merged = clientes_obl.merge(obl_stats, on=["num_doc","obl17"], how="left")

# Reconstrucción con recurrencia sobre todos los periodos
obl_merged["exclusivo_xperiodo"] = (
    pagos_unique.groupby(["num_doc","obl17"]).apply(
        lambda g: ((g["avg_pago_fisico_3m"].fillna(0) == 0) &
                   (g["avg_pago_virtual_3m"].fillna(0) == 0) &
                   (g["avg_pago_otros_3m"].fillna(0) == 0)).all()
    ).reset_index(drop=True) if False else None  # skip por complejidad
)

print(obl_merged.groupby("var_rta")["recurrencia_debito"].describe().round(3))
print()
print("  Recurrencia_sobre_total por clase:")
print(obl_merged.groupby("var_rta")["recurrencia_sobre_total"].describe().round(3))

# ── Umbral óptimo de recurrencia_sobre_total ──────────────────────────────────
print("\n── Agreement por umbral de recurrencia_sobre_total (cross-periodo) ──────")
for thr in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    pred = (obl_merged["recurrencia_sobre_total"] >= thr).astype(int)
    real = obl_merged["var_rta"]
    ag   = (pred == real).mean()
    tp   = ((pred==1) & (real==1)).sum()
    fp   = ((pred==1) & (real==0)).sum()
    fn   = ((pred==0) & (real==1)).sum()
    tn   = ((pred==0) & (real==0)).sum()
    print(f"  thr={thr:.1f}: agreement={ag:.4%}  TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")

# ── Distribución de periodos por obligación ───────────────────────────────────
print("\n── Periodos disponibles por obligación ────────────────────────────────")
print(obl_merged["periodos_total"].value_counts().sort_index())
