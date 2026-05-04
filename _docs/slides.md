---
marp: true
theme: default
paginate: true
style: |
  :root {
    --color-yellow: #FFD100;
    --color-blue:   #003087;
    --color-dark:   #1A1A2E;
    --color-gray:   #F5F5F5;
    --color-text:   #222222;
    --color-green:  #1B6B3A;
    --color-red:    #8B1C1C;
  }

  section {
    background: #FFFFFF;
    color: var(--color-text);
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 22px;
    padding: 40px 55px;
  }

  section.cover {
    background: var(--color-blue);
    color: #FFFFFF;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  section.cover h1 { font-size: 42px; color: var(--color-yellow); margin-bottom: 8px; }
  section.cover h2 { font-size: 22px; color: #FFFFFF; font-weight: 400; margin-top: 0; }
  section.cover p  { font-size: 16px; color: #CCCCCC; margin-top: 30px; }

  section.section-title {
    background: var(--color-yellow);
    color: var(--color-dark);
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  section.section-title h1 { font-size: 48px; margin: 0; }
  section.section-title p  { font-size: 20px; margin-top: 10px; color: #444; }

  section.insight {
    background: var(--color-dark);
    color: #FFFFFF;
  }
  section.insight h2 { color: var(--color-yellow); border-bottom: 3px solid var(--color-yellow); padding-bottom: 6px; }
  section.insight table th { background: #FFD100 !important; color: #1A1A2E !important; }
  section.insight table td { color: #FFFFFF !important; background: #1A1A2E !important; border-bottom: 1px solid #444; }
  section.insight table tr:nth-child(even) td { background: #2A2A3E !important; color: #FFFFFF !important; }
  section.insight pre { background: #0B0B1A !important; border: 1px solid #4A4A6A; color: #F8F8F2; }
  section.insight pre code { background: transparent !important; color: #F8F8F2 !important; }

  h2 { color: var(--color-blue); border-bottom: 3px solid var(--color-yellow); padding-bottom: 6px; }
  h3 { color: var(--color-blue); font-size: 20px; }

  table { width: 100%; border-collapse: collapse; font-size: 18px; }
  th { background: var(--color-blue); color: #FFF; padding: 8px 12px; }
  td { padding: 7px 12px; border-bottom: 1px solid #DDD; }
  tr:nth-child(even) td { background: var(--color-gray); }

  :not(pre) > code { background: #EEF2FF; color: #003087; padding: 2px 6px; border-radius: 4px; font-size: 16px; }
  pre      { background: #1A1A2E; color: #F8F8F2; padding: 16px 20px; border-radius: 8px; font-size: 15px; }
  pre code { background: transparent; color: #F8F8F2; padding: 0; border-radius: 0; font-size: 15px; }

  .pill-green  { background: #D4EDDA; color: #155724; padding: 2px 10px; border-radius: 12px; font-size: 16px; font-weight: 600; }
  .pill-yellow { background: #FFF3CD; color: #856404; padding: 2px 10px; border-radius: 12px; font-size: 16px; font-weight: 600; }
  .pill-red    { background: #F8D7DA; color: #721C24; padding: 2px 10px; border-radius: 12px; font-size: 16px; font-weight: 600; }
  .pill-blue   { background: #CCE5FF; color: #003087; padding: 2px 10px; border-radius: 12px; font-size: 16px; font-weight: 600; }

  .big-number { font-size: 52px; font-weight: 800; color: var(--color-yellow); }
  .big-label  { font-size: 16px; color: #CCCCCC; display: block; margin-top: -8px; }

  footer { font-size: 13px; color: #999; }
  section::after { font-size: 13px; color: #999; }
---

<!-- _class: cover -->

# Débitos Recurrentes
## Prueba Técnica — Analítico III

**Jhon Fredy Correa Gomez**
Economista · Científico de Datos

<p>Bancolombia · Gerencia de Inteligencia de Originación y Cobranza · 2026</p>

---

## Agenda

| # | Tema | Tiempo |
|---|------|--------|
| 1 | Contexto e impacto de negocio | 2 min |
| 2 | Arquitectura y pipeline escalable | 2 min |
| 3 | Hallazgos críticos del EDA | 2 min |
| 4 | Feature engineering y selección | 1 min |
| 5 | Modelos, resultados y diagnóstico AUC | 3 min |
| 6 | Dashboard Metabase | 2 min |
| 7 | Operacionalización con Airflow | 1 min |
| 8 | Conclusiones, accionables y próximos pasos | 2 min |

---

<!-- _class: insight -->

## Resumen ejecutivo — en 30 segundos

| Métrica | Valor |
|---------|-------|
| **AUC real del modelo (Gradient Boosting)** | **0.96** — Test; **0.96** — OOT |
| **KS estadístico** | **0.73** — excelente separación de clases |
| **Estabilidad** | Train ≈ Test ≈ OOT — sin sobreajuste |
| **Clase 1 (pagadores por débito)** | **78.8 %** del universo (36,835 obligaciones) |
| **Modelos desplegados** | XGBoost · Gradient Boosting · Logistic Regression |
| **Opción sin historial (B)** | AUC **0.88** (solo gestiones + moras, 19 features) |
| **Pipeline** | CSV → PostgreSQL → Parquets → PKL → Metabase — **100 % automatizado** |
| **Diagnóstico AUC = 1.0** | Identificado, explicado y resuelto — causa estructural del dataset |

> El modelo identifica con **AUC = 0.96** qué obligaciones en mora pagarán exclusivamente por débito recurrente, permitiendo **desactivar la gestión humana** en el 78 % de la cartera.

---

<!-- _class: section-title -->

# 1. Contexto
Problema de negocio e impacto operativo

---

## Problema de negocio

> **¿Qué se quiere resolver?**
> Identificar obligaciones en mora temprana (1–30 días) con **alta probabilidad de pagar exclusivamente por débito recurrente**, para reducir costos de cobranza y mejorar la eficiencia operativa.

### Definición de la variable respuesta

| Clase | Condición | Registros | % |
|-------|-----------|-----------|---|
| `var_rta = 1` | Pagos **únicamente** por débito **Y** recurrencia ≥ 40 % | 36,835 | **78.8 %** |
| `var_rta = 0` | Sin patrón de pago registrado en el período | 9,901 | **21.2 %** |

> **Condición doble simultánea:** exclusividad de canal + frecuencia mínima ≥ 40 %.
> **Hallazgo crítico:** clase 0 ≠ "pagó por otro canal" — significa **sin actividad de pago** en el período. Ver H1.

Desbalance **3.7:1** → estratificación + `class_weight` en todos los modelos.

---

## Segmentación operativa de cobranza

| Segmento | Criterio | Acción | Impacto |
|----------|----------|--------|---------|
| **A — Automatizar** | `var_rta=1` y mora ≤ 10 días | **Sin gestión humana** | Ahorro máximo de costos |
| **B — Monitorear** | `var_rta=1` y mora > 10 días | Seguimiento ligero | Reducción parcial de costos |
| **C — Cobranza suave** | `var_rta=0` y mora ≤ 15 días | Contacto preventivo | Gestión focalizada |
| **D — Cobranza intensiva** | `var_rta=0` y mora > 15 días | Gestión activa | Máximo esfuerzo |

> **Palanca clave:** el modelo predice `var_rta` con AUC = 0.96 → permite **concentrar el 100 % del esfuerzo humano en Segmentos C y D**, que representan el 21.2 % de la cartera.

---

## Impacto directo en costos de cobranza

El modelo permite **priorizar** qué obligaciones NO necesitan gestión humana:

```
Universo en mora temprana:  45,731 obligaciones
         │
         ├── Clase 1 predicha (AUC 0.96): ~36,000 obligaciones
         │       ├── Segmento A (mora ≤ 10d): automatizar → sin costo de gestor
         │       └── Segmento B (mora > 10d): monitoreo ligero
         │
         └── Clase 0 predicha: ~9,700 obligaciones → gestión focalizada C/D
```

**Sin el modelo:** gestión activa sobre las 45,731 obligaciones.
**Con el modelo:** gestión activa solo sobre ~9,700 (Segmentos C + D).

> **Reducción estimada de volumen de gestión:** ~79 % — dependiente del umbral operativo elegido (actualmente KS-óptimo).

---

<!-- _class: section-title -->

# 2. Arquitectura
Modelo de datos analítico y pipeline escalable

---

## Fuentes de datos — 6 archivos CSV

| Fuente | Filas | Cols | Contenido | Cobertura |
|--------|-------|------|-----------|-----------|
| `clientes` | 46,736 | 4 | Base maestra con `var_rta` | 100 % |
| `pagos` (tanque) | 47,194 | 67 | Pagos por canal: débito, físico, virtual | 100 % |
| `gestiones` | 47,194 | 83 | Gestiones de cobranza (acuerdos, RPC) | 100 % |
| `excedentes` | 47,194 | 35 | Pagos sobre cuota y % de excedente | 100 % |
| `moras` | 47,194 | 12 | Días de mora por ventana temporal | 100 % |
| `canales` | 10,801 | 4,258 | Transacciones granulares por canal | **20.7 %** |

**Clave primaria compuesta:** `(num_doc, obl17, f_analisis)`
**Decisión:** `canales` → LEFT JOIN + imputar 0 (S4); solo 3 cols de resumen para el modelo (S10)

---

## Pipeline — flujo de datos end-to-end

```
CSV datalake/  (6 fuentes · 245K+ filas)
    │
    ▼ T1: loader.py         → PostgreSQL debitdb (6 tablas + dedup + S11)
    │
    ▼ T2: data_preparation.py → analytical_model.parquet (JOIN 6 fuentes)
    │
    ▼ T3: feature_engineering.py → train / test / oot .parquet  (~240 features)
    │
    ▼ T3.5: feature_selection.py → feature_cols.json
    │       (varianza → ANOVA → ElasticNet — fit solo en train)
    │
    ▼ T4: train_debit_classifier.py → PKL + MLflow
    │     (XGBoost · GBM · LogReg · Optuna 5 trials · 12 figuras/run)
    │
    ▼ T5: export_*_metrics.py → debit_model_metrics + debit_pipeline_metrics
    │
    ▼ T6: metabase_provisioning.py → Dashboard id=4 (18 cards · idempotente)
```

**Orquestado con Airflow DAG** `@monthly` · cada tarea como `DockerOperator` sobre `debit-network` · gates `ShortCircuitOperator` configurables

---

## Infraestructura Docker — `debit_network`

```
┌──────────────────────────────── debit_network ────────────────────────────────┐
│                                                                                │
│  ┌────────────────────────────────────────────────────────────────────────┐   │
│  │                         PostgreSQL  :5432                              │   │
│  │      debitdb · mlflowdb · metabaseappdb · airflowdb                   │   │
│  │                         Vol: postgres-db                               │   │
│  └──────────┬──────────────────────┬────────────────────────┬────────────┘   │
│             │                      │                        │                │
│  ┌──────────┴──────────┐  ┌────────┴────────┐  ┌───────────┴──────────────┐  │
│  │      mlflow         │  │    metabase     │  │      Stack Airflow        │  │
│  │  debit-ml:latest    │  │  metabase/      │  │  debit-airflow:latest    │  │
│  │  :5000              │  │  :3000          │  │  webserver :8080         │  │
│  │  Vol: data/mlflow   │  │                 │  │  scheduler               │  │
│  └─────────────────────┘  └─────────────────┘  │  Vol: dags/ artifacts/   │  │
│                                                 └──────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────┘
```

| Imagen | Base | Propósito |
|--------|------|-----------|
| `debit-ml:latest` | Python 3.11 + UV | Pipeline ML, MLflow server, migraciones Alembic |
| `debit-airflow:latest` | `apache/airflow:2.8.1-python3.11` | Orquestación DAG + providers Docker/PostgreSQL |

---

## Particionamiento Train / Test / OOT

| Partición | Rango temporal | Periodos | Filas |
|-----------|---------------|----------|-------|
| **Train** | 2024-07 → 2025-02 | 8 | 22,293 |
| **Test** | 2025-03 → 2025-07 | 5 | 13,693 |
| **OOT** | 2025-08 → 2025-11 | 4 | 10,750 |

**Justificación del esquema:**
- Split temporal = validación rigurosa **fuera de muestra** en cohortes futuras
- OOT representa el comportamiento más reciente de la cartera
- Hallazgo H4: 97.9 % de obligaciones aparecen en un único `f_analisis` → dataset casi **transversal** → el split segmenta cohortes distintas (ver Sección 5)

**Métricas de evaluación:** AUC · KS · Precision / Recall / F1 · Brier Score · Log Loss · Calibración

---

<!-- _class: section-title -->

# 3. EDA
Hallazgos críticos — lo que los datos revelan

---

<!-- _class: insight -->

## Hallazgo H1 — El más importante del análisis

> **El 100 % de observaciones clase-0 tienen `total_pago = 0` en TODAS las ventanas y canales del tanque.**

| Variable (tanque) | Clase 0 (9,901 obs) | Clase 1 (36,835 obs) |
|-------------------|--------------------|--------------------|
| `avg_pago_debito_3m` | **0.00** — todas las obs | > 0 en pagadores |
| `avg_pago_fisico_3m` | **0.00** — todas las obs | 0 (exclusividad débito) |
| `avg_pago_virtual_3m` | **0.00** — todas las obs | 0 (exclusividad débito) |
| `avg_pago_otros_3m` | **0.00** — todas las obs | 0 (exclusividad débito) |

**Lo que esto cambia:**
- `var_rta = 0` **no significa** "pagó por otro canal" → significa **"sin actividad de pago"**
- El objetivo real: **identificar quién va a pagar vs. quién no paga**
- Las features del tanque discriminan casi perfectamente → AUC = 0.96
- Explica el AUC = 1.0 con split temporal (ver Sección 5)

---

## Hallazgos complementarios

| ID | Hallazgo | Implicación |
|----|----------|-------------|
| **H2** | 14.9 % de clase-1 tiene pagos físico/virtual en ventana 3m | `var_rta` se calcula en `f_analisis` puntual; tanque promedia 3m previos → pequeña discrepancia esperada |
| **H3** | `rec_e_v` tiene solo 239 obs activas (< 0.1 %) | No usable como feature — el tanque es la fuente correcta de señal de débito |
| **H4** | 97.9 % de obligaciones en un único `f_analisis` | Dataset casi transversal → split temporal segmenta **cohortes distintas**, no evolución de la misma obligación |

---

## Reconstrucción de `var_rta` desde el tanque

| Estrategia | Concordancia | FP | FN |
|------------|-------------|----|----|
| E1: avg_débito_3m > 0 + exclusividad | 65.70 % | **0** | 16,032 |
| E2: prop_débito_3m ≥ 40 % + exclusividad | 65.70 % | **0** | 16,032 |
| E3: avg_débito_12m > 0 + exclusividad | 55.82 % | **0** | 20,650 |
| E4: min_débito_3m > 0 + exclusividad | 65.38 % | **0** | 16,181 |

> **Cero Falsos Positivos** en todas las estrategias — el tanque 3m **nunca clasifica erróneamente un clase-0 como clase-1**.
> La baja concordancia refleja que `var_rta` usa una ventana puntual distinta al promedio histórico del tanque.
> **Conclusión:** la baja concordancia no es ruido — es una diferencia metodológica de ventanas temporales.

---

<!-- _class: section-title -->

# 4. Features
Ingeniería y selección supervisada

---

## Selección supervisada — 3 etapas, fit solo en train

```
~240 candidatas
    │
    ▼ Varianza / Sparsity (umbral 0.01)
  191 features  (−49: canales granulares + zeros constantes)
    │
    ▼ ANOVA F-test (percentil 80 — p-valor ajustado)
  152 features  (−39: pago_fisico y pago_otros ELIMINADOS → confirma H1)
    │
    ▼ ElasticNet L1/L2  (l1_ratio=0.7, C=0.1)
   60 features finales — Opción A (incluye débito)
   19 features finales — Opción B (solo gestiones + moras)
```

**Grupos finales (Opción A):** gestiones=48 · pagos=24 · derivadas=21 · canales_resumen=4 · moras=9 · excedentes=9

**Nota crítica:** `pago_fisico` y `pago_otros` eliminados por ANOVA → **confirma empíricamente** que clase-0 = sin pago en ningún canal (H1). La selección supervisada encontró sola el hallazgo del EDA.

---

<!-- _class: section-title -->

# 5. Modelos
Resultados, diagnóstico y decisión

---

## Resultados — Opción A (60 features, incluye débito del tanque)

| Modelo | Split | CV AUC | Test AUC | OOT AUC | KS Test |
|--------|-------|--------|----------|---------|---------|
| **Gradient Boosting** ⭐ | Aleatorio | **0.96** | **0.96** | **0.96** | **0.73** |
| XGBoost | Aleatorio | 0.96 | 0.96 | 0.96 | 0.73 |
| Logistic Regression | Aleatorio | 0.95 | 0.95 | 0.95 | 0.71 |
| Gradient Boosting | Temporal | 0.85 | ⚠️ 1.00 | ⚠️ 1.00 | — |
| XGBoost | Temporal | 0.84 | ⚠️ 1.00 | ⚠️ 1.00 | — |
| Logistic Regression | Temporal | 0.82 | ⚠️ 1.00 | ⚠️ 1.00 | — |

> ⚠️ **AUC = 1.0 temporal** = artefacto estructural (H1 + H4) — ver diagnóstico.
> **Producción:** split aleatorio · AUC real = **0.96** · Train ≈ Test ≈ OOT (sin sobreajuste).

---

## Resultados — Opción B (19 features, sin historial de débito)

| Modelo | Split | CV AUC | Test AUC | OOT AUC | Notas |
|--------|-------|--------|----------|---------|-------|
| **Gradient Boosting** ⭐ | Aleatorio | **0.89** | **0.89** | **0.88** | Mejor opción B |
| XGBoost | Aleatorio | 0.88 | 0.89 | 0.88 | Estable |
| Logistic Regression | Aleatorio | 0.87 | 0.88 | 0.87 | Baseline robusto |
| Gradient Boosting | Temporal | 0.75 | 0.92 | 0.90 | Inflación parcial — moras tienen señal temporal |
| XGBoost | Temporal | 0.75 | 0.93 | 0.91 | KS test = 0.79 |
| Logistic Regression | Temporal | 0.74 | 0.93 | 0.93 | KS test = 0.83 |

**Features Opción B:** solo `gestiones` (acuerdos, RPC, promesas, rank) + `moras` (avg/min/max/std × 3m/6m/9m)
**Excluidas:** 129 features de pagos · excedentes · canales · derivadas de débito

> **Nota sobre temporal B:** sin features de pago el efecto H1 no aplica directamente, pero las features de moras también tienen señal temporal → test/OOT AUC (~0.92–0.93) supera el CV (~0.74–0.75). El split aleatorio (~0.88–0.89) es el referente de producción.
> **Valor de Opción B:** AUC = 0.88 sin ningún dato de pago demuestra que el **comportamiento de cobranza per se** (gestiones, mora) tiene alto poder predictivo.

---

## Diagnóstico — ¿Por qué AUC = 1.0 con split temporal?

**Síntoma observado:**

| Split | CV Train | Test | OOT |
|-------|----------|------|-----|
| Temporal (Opción A) | 0.84 | ⚠️ **1.00** | ⚠️ **1.00** |
| **Aleatorio (Producción)** | **0.96** | **0.96** | **0.96** |

**Causa:** NO es leakage temporal clásico — las ventanas de features son **previas** a `f_analisis`.

**Causa real — interacción de dos hechos estructurales:**

| Factor | Efecto |
|--------|--------|
| **H1:** Clase-0 tiene `total_pago = 0` en todo el tanque | Features de débito discriminan trivialmente a clase-0 |
| **H4:** Dataset casi transversal (97.9 % en único período) | Split cronológico concentra clase-0 en cohortes de Test/OOT |

→ En Test/OOT del split temporal solo hay clase-0 sin ningún pago → **separación perfecta trivial**.

---

## Decisión — Opciones A y B implementadas

| Opción | Features | AUC real | Uso en producción |
|--------|----------|----------|-------------------|
| **A ✅ — Producción** | 60 features (incluye débito del tanque) | **0.96** | Cartera activa con historial de pagos |
| **B ✅ — Disponible** | 19 features (gestiones + moras únicamente) | **0.88** | Clientes nuevos o sin historial de débito |

**Opción A — justificación:** el historial de débito (`pago_debito_3m`) **está disponible en producción** para obligaciones en mora temprana. No es leakage. AUC = 0.96 es el valor real y reproducible.

**Opción B — valor:** AUC = 0.88 demuestra que el **comportamiento de cobranza per se** (gestiones, RPC, mora) tiene poder predictivo independiente del historial de pagos. Útil para ampliar cobertura.

> Ambas opciones entrenadas y rastreadas en MLflow (`feature_set = A / B`). 4 combinaciones disponibles: A/B × random/temporal.

---

<!-- _class: section-title -->

# 6. Dashboard
18 cards · Metabase · `localhost:3000/dashboard/4`

---

## Tablero — visión analítica y operativa

**Cards de negocio (1–7):** comportamiento de la cartera

| # | Card | Insight que expone |
|---|------|-------------------|
| 1 | Evolución % Débito Exclusivo por Período | Tendencia de clase 1 en el tiempo |
| 2 | Volumen de Obligaciones por Período | Cohortes y concentración temporal |
| 3 | Segmentos A/B/C/D — Distribución Global | Qué fracción de la cartera es automatizable |
| 4 | Mix de Canales de Pago por Clase | Diferencias comportamentales entre clases |
| 5 | Efectividad de Gestiones por Clase | Qué gestiones predicen el pago |
| 7 | KPI Resumen — Último Período | Dashboard ejecutivo de un vistazo |

**Cards de modelos (8–10):** comparación y explicabilidad

| # | Card | Insight que expone |
|---|------|-------------------|
| 8 | Comparativa AUC/KS por Modelo | Qué modelo rendir mejor en cada configuración |
| 9 | Estabilidad Train/Test/OOT | Ausencia de sobreajuste — confianza en producción |
| 10 | Top 20 Features SHAP | Qué variables explican el modelo |

**Cards de pipeline (11–18):** trazabilidad y monitoreo del proceso

---

<!-- _class: section-title -->

# 7. Operacionalización
DAG Airflow — ejecución programada y controlada

---

## DAG `debit_ml_pipeline` — arquitectura de ejecución

```
gate_setup         ──► setup_artifacts_dir
gate_load_data     ──► load_data          (CSV → PostgreSQL · dedup · S11)
gate_build_model   ──► build_analytical_model  (JOIN 6 fuentes · 45,731 obs)
gate_build_features ──► build_features     (~240 features → parquets)
gate_select_features ──► select_features   (60 / 19 features · feature_cols.json)
gate_train         ──► train_model         (XGBoost + GBM + LogReg + MLflow)
gate_metabase      ──► provision_metabase  (18 cards · idempotente)
```

**Cada gate:** `ShortCircuitOperator` configurable desde `.env` o al disparar el DAG
**Cada tarea:** `DockerOperator` sobre `debit-network` → **reproducible en cualquier entorno**
**Calendarizable:** `@monthly` por defecto · parámetros `split_strategy` × `feature_set` por run

```bash
make airflow-trigger   # pipeline completo
# Ejecución parcial desde UI: Trigger DAG → checkboxes por tarea
```

---

<!-- _class: section-title -->

# 8. Conclusiones
Requerimientos cumplidos · accionables · próximos pasos

---

## Requerimientos de la prueba — estado de cumplimiento

| Requerimiento | Solución implementada | Resultado |
|---------------|----------------------|-----------|
| **R1** Modelo de datos analítico | 6 tablas ORM SQLModel · clave `(num_doc, obl17, f_analisis)` · 5 migraciones Alembic · 9 vistas SQL | ✅ |
| **R2** Pipeline escalable | Airflow DAG `@monthly` · 7 tareas `DockerOperator` · gates configurables · `make pipeline` | ✅ |
| **R3** Split Train/Test/OOT | 8/5/4 períodos · justificación temporal documentada · diagnóstico AUC=1.0 identificado y resuelto | ✅ |
| **R4** Desempeño del modelo | AUC = 0.96 · KS = 0.74 · Precision/Recall/F1 · Brier · Log Loss · Calibración · SHAP | ✅ |
| **R5** Tablero análisis descriptivo | Metabase dashboard id=4 · 18 cards · EDA + modelos + pipeline · accionables de negocio | ✅ |

---

<!-- _class: insight -->

## Accionables inmediatos para el negocio

| Prioridad | Accionable | Impacto |
|-----------|-----------|---------|
| 🔴 Alta | **Activar Segmento A en producción** — desactivar gestión humana en obligaciones clase 1 con mora ≤ 10 días | Reducción de costos de gestión en ~79 % del volumen |
| 🔴 Alta | **Umbral operativo ajustado** — mover punto de corte según costo de falso negativo vs. costo de gestión | Maximizar ahorro neto |
| 🟡 Media | **Despliegue Opción B** para clientes sin historial de débito — AUC 0.88 amplía cobertura del modelo | Mayor cobertura sin degradar calidad |
| 🟡 Media | **Monitoreo Segmento B** — alerta automática cuando mora > 10 días en obligaciones clase 1 predicha | Prevención de deterioro |
| 🟢 Normal | **Re-entrenamiento mensual** via Airflow con monitoreo PSI sobre `prop_debito_3m` y score | Estabilidad del modelo en producción |
| 🟢 Normal | **Trigger automático** si AUC OOT < 0.90 en el próximo run mensual | Gobernanza del modelo |

---

## Conclusiones técnicas clave

| Conclusión | Evidencia |
|------------|-----------|
| **El modelo es válido y robusto** | AUC = 0.96, KS = 0.73, Train ≈ Test ≈ OOT sin sobreajuste |
| **H1 es el hallazgo más importante** | Clase-0 = sin pago — redefine el objetivo del modelo y explica el AUC inflado |
| **La selección supervisada confirma el EDA** | ANOVA elimina `pago_fisico` y `pago_otros` automáticamente → ambas fuentes convergen |
| **El diagnóstico AUC=1.0 es transparente** | Causa estructural identificada, documentada y resuelta — no es un error, es un insight |
| **La solución es end-to-end y productizable** | CSV → PostgreSQL → MLflow → Metabase · Airflow → re-entrenamiento mensual automatizado |
| **Opción B agrega valor** | AUC = 0.88 sin historial de débito — el comportamiento de cobranza per se predice el pago |

---

## Próximos pasos

1. **Umbral operativo:** ajustar punto de corte (actualmente KS-óptimo) mediante análisis costo-beneficio de gestión vs. falso negativo con el equipo de cobranza
2. **Monitoreo de drift:** PSI mensual sobre `prop_debito_3m` (top feature SHAP) y score del modelo — Airflow DAG ya calendarizado
3. **Despliegue dual A/B:** orquestar scoring con Opción A para cartera activa y Opción B para clientes sin historial, maximizando cobertura
4. **Re-entrenamiento automático:** trigger en DAG si AUC OOT < 0.90 en run mensual — gobernanza del modelo sin intervención manual
5. **Convergencia LogReg:** aumentar `max_iter` en `LogisticRegressionDebitClassifier` para resolver `ConvergenceWarning` (ALERTA 3)

---

<!-- _class: cover -->

# ¡Gracias!

**Jhon Fredy Correa Gomez**

Dashboard → `http://localhost:3000/dashboard/4`
MLflow → `http://localhost:5000` · Airflow → `http://localhost:8080`

<p>jonfredi12@gmail.com</p>
