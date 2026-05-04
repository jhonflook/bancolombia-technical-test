# 🏦 Débitos Recurrentes — Bancolombia

> **Prueba Técnica · Cargo: Analítico III**
>
> 👤 **Jhon Fredy Correa Gomez** · Economista · Científico de Datos
> 📧 jonfredi12@gmail.com · 
> 🔗 [linkedin.com/in/jhoncorrgo](https://www.linkedin.com/in/jhoncorrgo/)

📎 **Documentos del proyecto:**
&nbsp;&nbsp;📄 [Diagrama de Arquitectura (PDF)](./_docs/debit_architecture_diagram.pdf)
&nbsp;&nbsp;·&nbsp;&nbsp;🗂️ [Fuente Mermaid](./_docs/debit_architecture_diagram.md)
&nbsp;&nbsp;·&nbsp;&nbsp;📓 [Análisis Exploratorio](./notebooks/analisis_fuentes_debitos_recurrentes.ipynb)

---

## 📋 Contexto de Negocio

Cartera de clientes en **mora temprana (1–30 días)**. El objetivo es identificar obligaciones con alta probabilidad de pagarse **exclusivamente por débito recurrente**, reduciendo así los costos de cobranza.

| Clase | Condición |
|-------|-----------|
| `var_rta = 1` ✅ | Pagos **únicamente** por débito **Y** recurrencia ≥ 40 % |
| `var_rta = 0` ❌ | Cualquier pago por otro canal, o sin patrón de pago registrado |

### 📊 Distribución del Target

| Clase | Registros | % |
|-------|-----------|---|
| 1 — Débito exclusivo | 36,835 | 78.8 % |
| 0 — Sin patrón / otro canal | 9,901 | 21.2 % |

Desbalance **3.7 : 1** → estratificación + `class_weight`. Métrica principal: **AUC-ROC + KS + Precision/Recall**.

---

## 🏗️ Arquitectura

```
datalake/ (6 CSVs)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                   Docker — debit_network                │
│                                                         │
│  ┌──────────────┐   ┌──────────────┐   ┌─────────────┐ │
│  │  PostgreSQL  │   │    MLflow    │   │  Metabase   │ │
│  │   :5432      │   │   :5000      │   │   :3000     │ │
│  └──────────────┘   └──────────────┘   └─────────────┘ │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │        Apache Airflow 2.8.1 — :8080             │   │
│  │  DAG: debit_ml_pipeline (@monthly, 7 tareas)    │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

| 📄 Documento | 🔗 Link |
|---|---|
| Diagrama de Arquitectura (PDF) | [**Ver PDF →**](./_docs/debit_architecture_diagram.pdf) |
| Fuente Mermaid (8 diagramas) | [debit_architecture_diagram.md](./_docs/debit_architecture_diagram.md) |

---

## 🗄️ Fuentes de Datos

| Alias | Archivo | Filas | Cols | Descripción |
|-------|---------|-------|------|-------------|
| `clientes` | `*_clientes_seleccionados_prueba.csv` | 46,736 | 4 | Base maestra con `var_rta` |
| `canales` | `*_canales_*_prueba_tecnica_1.csv` | 10,801 | 4,258 | Transacciones por canal |
| `excedentes` | `*_excedente_*.csv` | 47,194 | 35 | Excedentes y % pagado sobre cuota |
| `gestiones` | `*_gestiones_*.csv` | 47,194 | 83 | Gestiones de cobranza |
| `moras` | `*_moras_*.csv` | 47,194 | 12 | Días de mora por ventana |
| `pagos` | `*_tanque_*.csv` | 47,194 | 67 | Pagos por canal (débito, físico, virtual) |

**Clave primaria compuesta:** `(num_doc, obl17, f_analisis)`
**Join:** `clientes → LEFT JOIN` excedentes, gestiones, moras, pagos, canales

---

## 🤖 Modelos y Resultados

### ✅ Modelo de Producción — Split Aleatorio (2026-05-02)

> El dataset es casi transversal (97.9 % de obs en un único período). El split aleatorio es la evaluación más representativa.

| Modelo | CV AUC | Train AUC | Train KS | Test AUC | Test KS | OOT AUC |
|--------|--------|-----------|----------|----------|---------|---------|
| **Gradient Boosting** ⭐ | **0.9616** | **0.9673** | **0.7579** | **0.9620** | **0.7379** | **0.9616** |
| XGBoost | 0.9587 | 0.9607 | 0.7324 | 0.9594 | 0.7300 | 0.9580 |
| Logistic Regression | 0.9548 | 0.9547 | 0.7158 | 0.9537 | 0.7100 | 0.9530 |

> 💡 `HistGradientBoostingClassifier` · Optuna 5 trials · 60 features seleccionadas · PKLs en `data/artifacts/`

### 🔍 Selección de Features — Pipeline Supervisado

```
240 candidatas
  → 191  (-49 varianza/sparsity)
  → 152  (-39 ANOVA F-test)
  → 60   (-92 ElasticNet L1=0.7, C=0.1)
```

Grupos finales: `gestiones` · `pagos` · `derivadas` · `canales_resumen` · `moras` · `excedentes`

### 🗂️ Split Temporal — Cohortes

| Partición | Rango | Filas |
|-----------|-------|-------|
| Train | 2024-07 → 2025-02 | 22,291 |
| Test | 2025-03 → 2025-07 | 13,712 |
| OOT | 2025-08 → 2025-11 | 10,733 |

---

## 🎯 Segmentación de Cobranza

| Segmento | Criterio | Acción |
|----------|----------|--------|
| 🟢 **A — Automatizar** | `var_rta=1` y `mora_6m ≤ 10 días` | Débito automático sin intervención |
| 🟡 **B — Monitorear** | `var_rta=1` y `mora_6m > 10 días` | Seguimiento preventivo |
| 🟠 **C — Cobranza suave** | `var_rta=0` y `mora_6m ≤ 15 días` | Gestión ligera |
| 🔴 **D — Cobranza intensiva** | `var_rta=0` y `mora_6m > 15 días` | Gestión activa |

### Criterio de los umbrales — justificación operativa

Los umbrales de `mora_6m` son **supuestos de diseño operativo** (complementan S8), no derivados estadísticamente. Se sustentan en la naturaleza de la cartera (mora temprana 1–30 días) y en el diferente costo de error según el segmento:

**¿Por qué 10 días para los segmentos A y B (clase 1)?**

`mora_6m` es el promedio de días de mora en los últimos 6 meses. Un cliente con `var_rta=1` (paga exclusivamente por débito, recurrencia ≥ 40 %) y `mora_6m ≤ 10 días` muestra que sus retrasos históricos son **incidentales** — fondos momentáneamente insuficientes o delays bancarios, no un problema estructural de capacidad de pago. La combinación canal automático + historial de mora bajo hace que la recuperación sea prácticamente segura sin intervención humana.

El umbral es **intencionalmente estricto en 10 días** (no 15) porque el segmento A elimina completamente la gestión humana: un error tiene costo operativo alto. Los clientes con `var_rta=1` y mora entre 10–30 días siguen teniendo el canal correcto, pero su historial de mora más alto sugiere que el débito por sí solo puede no ser suficiente → se monitoran preventivamente (segmento B).

**¿Por qué 15 días para los segmentos C y D (clase 0)?**

La clase 0 equivale a **sin actividad de pago** en el período (Hallazgo H1) — todos estos clientes requieren gestión humana. El umbral de 15 días separa mora "temprana leve" (gestión suave, posible recuperación rápida) de mora más avanzada dentro del rango 1–30 días (gestión intensiva). El corte es más permisivo que 10 días porque aquí la variable relevante es la **intensidad** de la gestión, no si se gestiona o no.

> **Resumen del diseño:** el umbral de 10 días para clase 1 maximiza la certeza antes de desactivar la gestión humana; el umbral de 15 días para clase 0 calibra el esfuerzo entre gestión suave e intensiva.

---

## 📈 Dashboard Metabase

**URL:** `http://localhost:3000/dashboard/4`
**Credenciales:** `admin@bancolombia.com` / `Debit2026!Bancolombia`

| # | Card | Tipo | Fuente |
|---|------|------|--------|
| 1 | Evolución % Débito Exclusivo por Período | 📉 Línea | `v_debit_kpi_period` |
| 2 | Volumen de Obligaciones por Período | 📊 Barras | `v_debit_kpi_period` |
| 3 | Segmentos de Cobranza — Distribución Global | 🥧 Pie | `v_debit_risk_segment_summary` |
| 4 | Mix de Canales de Pago por Clase | 📊 Barras | `v_debit_payment_mix` |
| 5 | Efectividad de Gestiones por Clase | 📊 Barras | `v_debit_gestiones_profile` |
| 6 | Segmentos de Cobranza por Período | 📊 Barras | `v_debit_risk_segment_summary` |
| 7 | KPI Resumen — Último Período | 🔢 Escalar | `v_debit_kpi_period` |
| 8 | Comparativa AUC y KS por Modelo | 📋 Tabla | `debit_model_metrics` |
| 9 | Estabilidad Train vs Test vs OOT | 📊 Barras | `debit_model_metrics` |
| 10 | Top 20 Features por Modelo | 📊 Barras | `debit_model_features` |

---

## 🚀 Inicio Rápido

### 1️⃣ Prerrequisitos

- 🐳 Docker + Docker Compose
- 🐍 Python ≥ 3.12 · [uv](https://docs.astral.sh/uv/)
- Datos en `datalake/` (6 archivos CSV)

### 2️⃣ Configurar entorno

```bash
# Activar uv
source /home/jhonflook/snap/code/237/.local/bin/env

# Construir imágenes Docker
make build-all        # debit-ml:latest + debit-airflow:latest
```

### 3️⃣ Levantar servicios base

```bash
make up               # PostgreSQL + MLflow
make migrate          # Alembic: tablas + vistas
```

### 4️⃣ Ejecutar pipeline completo

**Opción A — Makefile directo (sin Airflow):**

```bash
make pipeline         # Carga → modelo analítico → features → selección → entrenamiento
```

**Opción B — DAG Airflow:**

```bash
make airflow-init     # Primera vez: inicializa BD + crea usuario admin
make airflow-up       # Levanta webserver + scheduler
make airflow-trigger  # Dispara debit_ml_pipeline manualmente
make airflow-status   # Monitorea el run
```

### 5️⃣ Exportar métricas y levantar dashboard

```bash
make export-model-metrics-overwrite   # MLflow → debit_model_metrics + features
make export-shap-features-overwrite   # SHAP → gradient_boosting features
make up-core                          # Levanta Metabase
make provision-metabase               # Sincroniza dashboard (idempotente)
```

### 6️⃣ Generar diagrama de arquitectura en PDF

```bash
make diagram-pdf      # → _docs/debit_architecture_diagram.pdf
```

---

## 🛠️ Stack Tecnológico

| Capa | Tecnología | Versión |
|------|------------|---------|
| 🗄️ Base de datos | PostgreSQL | 17 |
| 🔄 Orquestación | Apache Airflow | 2.8.1 |
| 📊 Visualización | Metabase | latest |
| 🧪 Experiment tracking | MLflow | ≥ 3.8 |
| ⚡ Optimización | Optuna | ≥ 4.0 |
| 🤖 ML — Boosting | XGBoost + HistGBM | ≥ 3.2 / sklearn |
| 📐 ML — Baseline | Logistic Regression | sklearn ≥ 1.6 |
| 🔍 Interpretabilidad | SHAP (TreeExplainer) | ≥ 0.46 |
| 🐍 Runtime | Python | ≥ 3.12 |
| 📦 Paquetes | uv | latest |
| 🐳 Contenedores | Docker Compose | v2 |
| 🗂️ ORM | SQLModel + Alembic | — |

---

## 📁 Estructura del Proyecto

```
📦 bancolombia-technical-test/
├── 🐳 deploy/
│   ├── Dockerfile                   # imagen debit-ml:latest
│   ├── Dockerfile.airflow           # imagen debit-airflow:latest
│   ├── entrypoint.sh                # alembic upgrade + cmd
│   ├── train_debit_classifier.py    # entrypoint ML multi-modelo
│   ├── export_model_metrics.py      # MLflow → DB métricas
│   ├── export_shap_features.py      # SHAP → gradient_boosting features
│   └── metabase_provisioning.py     # provisiona 10 cards (idempotente)
│
├── 📂 src/
│   ├── settings.py                  # Pydantic Settings
│   ├── database/                    # engine, get_session(), CRUD
│   ├── dataset/
│   │   ├── loader.py                # CSV → PostgreSQL (streaming)
│   │   ├── data_preparation.py      # LEFT JOIN → analytical_model.parquet
│   │   ├── feature_engineering.py   # ~240 features candidatas + split
│   │   ├── feature_selection.py     # varianza → ANOVA → ElasticNet
│   │   └── split_strategies.py      # temporal (prod) / random (diag)
│   ├── models/debit.py              # 6 SQLModel tables
│   ├── services/
│   │   ├── training.py              # run_training_pipeline()
│   │   ├── forecasting.py           # DebitScoringService → A/B/C/D
│   │   └── optimization.py          # ModelSelectionService
│   └── statistical_models/          # XGBoost · HistGBM · LogReg · RF (base ABC)
│
├── 📂 data/
│   ├── dags/debit_ml_pipeline_dag.py  # DAG @monthly, 7 tareas + gates
│   ├── artifacts/                     # train/test/oot.parquet · model_*.pkl
│   └── queries/debit_views.sql        # 9 vistas v_debit_*
│
├── 📂 _docs/
│   ├── debit_architecture_diagram.md   # fuente Mermaid (8 diagramas)
│   ├── debit_architecture_diagram.html # render intermedio
│   └── debit_architecture_diagram.pdf  # ← diagrama exportado
│
├── 📂 notebooks/
│   └── analisis_fuentes_debitos_recurrentes.ipynb
│
├── 📂 alembic/versions/
│   ├── c1d2e3f4a5b6  tablas debit_*         ✅
│   ├── d2e3f4a5b6c7  vistas v_debit_*       ✅
│   └── e3f4a5b6c7d8  debit_model_metrics    ✅
│
├── 🐳 docker-compose.yml
├── ⚙️ makefile
├── 📋 pyproject.toml
└── 🔑 .env
```

> 📄 **Documentación:** [Diagrama de Arquitectura PDF](./_docs/debit_architecture_diagram.pdf) · [Fuente Mermaid](./_docs/debit_architecture_diagram.md) · [Notebook EDA](./notebooks/analisis_fuentes_debitos_recurrentes.ipynb)

---

## ⚙️ Comandos Makefile — Referencia Rápida

```bash
make help                      # lista todos los targets

# 🐳 Imágenes
make build-ml                  # debit-ml:latest
make build-airflow             # debit-airflow:latest
make build-all                 # ambas

# 🏗️ Infraestructura
make up                        # postgres + mlflow
make up-core                   # + metabase
make down                      # detiene todo

# 🔄 Airflow
make airflow-init              # primera vez
make airflow-up / airflow-down
make airflow-trigger           # dispara DAG
make airflow-status            # lista runs

# 📥 Datos
make load                      # todos los CSVs
make build-model               # analytical_model.parquet

# 🔧 Features
make features                  # split temporal (producción)
make features-random           # split aleatorio (diagnóstico)
make select-features           # varianza → ANOVA → ElasticNet

# 🤖 Entrenamiento
make train                     # 3 modelos, 5 trials
make train-random-split        # diagnóstico Alerta 1

# 🚀 Pipeline completo
make pipeline                  # T1 → T2 → T3 → T3.5 → T4
make pipeline-from-features    # T3 → T3.5 → T4

# 📊 Metabase
make export-model-metrics-overwrite
make export-shap-features-overwrite
make provision-metabase

# 📄 Documentación
make diagram-pdf               # genera PDF desde HTML+Mermaid
```

---

## 🔬 Hallazgos Clave

| # | Hallazgo |
|---|----------|
| H1 | **Clase 0 = sin actividad de pago**: 100 % de obs clase-0 tienen `total_pago = 0` en todas las ventanas |
| H2 | **`var_rta` se calcula sobre el período puntual** de `f_analisis`; el tanque promedia los 3m previos |
| H3 | **`rec_e_v` no es señal del target**: solo 239 obs activas (< 0.1 %) |
| H4 | **Dataset casi transversal**: 97.9 % de obligaciones aparece en un único `f_analisis` |

> ⚠️ **Alerta 1 — Resuelta:** el AUC = 1.0 con split temporal era un artefacto estructural (clase-0 sin pagos + cohortes separadas), no sobreajuste. AUC real = **~0.96** con split aleatorio.

---

<div align="center">

**Jhon Fredy Correa Gomez** · Prueba Técnica Analítico III · Bancolombia · 2026

</div>
