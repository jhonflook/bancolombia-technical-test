"""
Verificación del cálculo de var_rta contra los datos de pagos (tanque).

Hipótesis a corroborar:
  var_rta = 1  ⟺  pagos exclusivamente por débito (fisico=0, virtual=0, otros=0)
                   Y recurrencia_debito >= 40%

Usamos la ventana de 3m como referencia principal (menor ruido por historial corto).
"""
import pandas as pd
import numpy as np

CLIENTES = "datalake/clientes_seleccionados_prueba.csv"
PAGOS    = "datalake/debitos_dev_cruce_tanque_poblacion_prueba_tecnica.csv"
JOIN_KEYS = ["num_doc", "obl17", "f_analisis"]

print("Cargando archivos...")
clientes = pd.read_csv(CLIENTES, dtype={k: str for k in JOIN_KEYS})
pagos    = pd.read_csv(PAGOS,    dtype={k: str for k in JOIN_KEYS})

# Deduplicar pagos (copias exactas documentadas en S11)
pagos = pagos.drop_duplicates()

df = clientes.merge(pagos, on=JOIN_KEYS, how="left")
print(f"Filas post-join: {len(df):,}  |  target 1: {df.var_rta.sum():,}  |  target 0: {(df.var_rta==0).sum():,}")

# ── Análisis por ventana ─────────────────────────────────────────────────────
results = {}
for w in ["3m", "6m", "9m", "12m"]:
    d  = df[f"avg_pago_debito_{w}"].fillna(0)
    f_ = df[f"avg_pago_fisico_{w}"].fillna(0)
    v  = df[f"avg_pago_virtual_{w}"].fillna(0)
    o  = df[f"avg_pago_otros_{w}"].fillna(0)

    total = d + f_ + v + o
    prop_debito = np.where(total > 0, d / total, 0.0)

    # Condición A: exclusividad (fisico=0 AND virtual=0 AND otros=0)
    exclusivo = (f_ == 0) & (v == 0) & (o == 0)
    # Condición B: recurrencia >= 40%
    recurrente = prop_debito >= 0.40
    # var_rta reconstruida
    var_rta_rec = (exclusivo & recurrente).astype(int)

    agreement  = (var_rta_rec == df["var_rta"]).mean()
    tp = ((var_rta_rec == 1) & (df["var_rta"] == 1)).sum()
    fp = ((var_rta_rec == 1) & (df["var_rta"] == 0)).sum()
    fn = ((var_rta_rec == 0) & (df["var_rta"] == 1)).sum()
    tn = ((var_rta_rec == 0) & (df["var_rta"] == 0)).sum()
    results[w] = dict(agreement=agreement, tp=tp, fp=fp, fn=fn, tn=tn,
                      pred1=var_rta_rec.sum(), real1=df["var_rta"].sum())

print("\n── Concordancia var_rta vs reconstrucción por ventana ──────────────────")
print(f"{'Ventana':<8} {'Concordancia':>12} {'TP':>8} {'FP':>8} {'FN':>8} {'TN':>8} {'Pred1':>8} {'Real1':>8}")
for w, r in results.items():
    print(f"{w:<8} {r['agreement']:>11.4%} {r['tp']:>8,} {r['fp']:>8,} {r['fn']:>8,} {r['tn']:>8,} {r['pred1']:>8,} {r['real1']:>8,}")

# ── Análisis de discordancias en ventana 3m ──────────────────────────────────
w = "3m"
d  = df[f"avg_pago_debito_{w}"].fillna(0)
f_ = df[f"avg_pago_fisico_{w}"].fillna(0)
v  = df[f"avg_pago_virtual_{w}"].fillna(0)
o  = df[f"avg_pago_otros_{w}"].fillna(0)
total      = d + f_ + v + o
prop_debito = np.where(total > 0, d / total, 0.0)
exclusivo  = (f_ == 0) & (v == 0) & (o == 0)
recurrente = prop_debito >= 0.40
var_rta_rec = (exclusivo & recurrente).astype(int)

discordantes = df[var_rta_rec != df["var_rta"]].copy()
discordantes["prop_debito_3m"] = prop_debito[discordantes.index]
discordantes["exclusivo_3m"]   = exclusivo[discordantes.index]
discordantes["recurrente_3m"]  = recurrente[discordantes.index]
discordantes["var_rta_rec"]    = var_rta_rec[discordantes.index]

print(f"\n── Discordancias ventana 3m: {len(discordantes):,} filas ──────────────────────")
print("\nFalsos Positivos (rec=1, real=0):")
fp_df = discordantes[discordantes["var_rta_rec"] == 1]
print(fp_df[["var_rta","var_rta_rec","prop_debito_3m","exclusivo_3m","recurrente_3m",
             f"avg_pago_debito_{w}", f"avg_pago_fisico_{w}",
             f"avg_pago_virtual_{w}", f"avg_pago_otros_{w}"]].head(10).to_string())

print("\nFalsos Negativos (rec=0, real=1):")
fn_df = discordantes[discordantes["var_rta_rec"] == 0]
print(fn_df[["var_rta","var_rta_rec","prop_debito_3m","exclusivo_3m","recurrente_3m",
             f"avg_pago_debito_{w}", f"avg_pago_fisico_{w}",
             f"avg_pago_virtual_{w}", f"avg_pago_otros_{w}"]].head(10).to_string())

# ── Distribución de prop_debito por clase ────────────────────────────────────
print("\n── Distribución de prop_debito_3m por clase real ──────────────────────")
df["prop_debito_3m"] = prop_debito
print(df.groupby("var_rta")["prop_debito_3m"].describe().round(4))

# ── Casos con total_pago=0 ───────────────────────────────────────────────────
sin_pagos = (total == 0)
print(f"\nObligaciones sin actividad de pagos en 3m: {sin_pagos.sum():,}")
print(df[sin_pagos].groupby("var_rta").size().rename("count"))
