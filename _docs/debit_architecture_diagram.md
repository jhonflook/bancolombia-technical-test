# Diagrama de Arquitectura — Débitos Recurrentes (Bancolombia)

## Arquitectura General

```mermaid
flowchart TB
    subgraph External["Fuentes de Datos"]
        Datalake["datalake/<br/>6 archivos CSV"]
    end

    subgraph Docker["Infraestructura Docker"]
        subgraph Airflow["Apache Airflow"]
            Webserver["Airflow Webserver<br/>:8080"]
            Scheduler["Airflow Scheduler"]
            Init["Airflow Init"]
        end

        subgraph Storage["Almacenamiento"]
            Postgres["PostgreSQL :5432"]
            subgraph Databases["Bases de datos"]
                DebitDB["debitdb"]
                AirflowDB["airflow"]
                MetabaseAppDB["metabaseappdb"]
                MLflowDB["mlflowdb"]
            end
        end

        DebitML["debit-ml:latest<br/>Container"]
        MetabaseSvc["Metabase<br/>:3000"]
        MLflowSvc["MLflow Server<br/>:5000"]
    end

    subgraph Pipeline["DAG — debit_ml_pipeline (@monthly)"]
        T1["T1 load_data"]
        T2["T2 build_analytical_model"]
        T3["T3 build_features"]
        T35["T3.5 select_features"]
        T4["T4 train_model"]
        T5["T5 provision_metabase"]
    end

    Datalake --> T1
    T1 --> DebitML
    T2 --> DebitML
    T3 --> DebitML
    T35 --> DebitML
    T4 --> DebitML
    DebitML --> DebitDB
    DebitML --> MLflowSvc
    T5 --> MetabaseSvc
    MetabaseSvc --> DebitDB
    MLflowSvc --> MLflowDB

    Scheduler --> Pipeline
    Init --> Webserver
    Webserver --> Scheduler
    Postgres --> Databases
```

## Flujo del Pipeline de Datos

```mermaid
flowchart LR
    subgraph Ingest["Ingesta de Datos"]
        CSV1["clientes_seleccionados.csv"]
        CSV2["canales_prueba_tecnica.csv"]
        CSV3["excedente.csv"]
        CSV4["gestiones.csv"]
        CSV5["moras.csv"]
        CSV6["tanque.csv (pagos)"]
        Loader["loader.py<br/>streaming 2-pasadas"]
    end

    subgraph DBTables["debitdb — Tablas"]
        TClient["debit_client"]
        TPago["debit_pago"]
        TGestion["debit_gestion"]
        TMora["debit_mora"]
        TExcedente["debit_excedente"]
        TCanal["debit_canal"]
    end

    subgraph Preparation["Preparación Analítica"]
        DataPrep["data_preparation.py<br/>build_analytical_model()"]
        AnalyticalModel["analytical_model.parquet"]
    end

    subgraph Features["Feature Engineering"]
        FeatEng["feature_engineering.py<br/>build_debit_features()"]
        TrainParq["train.parquet<br/>(22,291 filas)"]
        TestParq["test.parquet<br/>(13,712 filas)"]
        OOTParq["oot.parquet<br/>(10,733 filas)"]
    end

    subgraph Selection["Selección de Features"]
        FeatSel["feature_selection.py<br/>varianza → ANOVA → ElasticNet"]
        FeatCols["feature_cols.json<br/>(60 features — Opción A)"]
    end

    subgraph Training["Entrenamiento ML"]
        TrainScript["train_debit_classifier.py"]
        MLflowRuns["MLflow Runs<br/>(debit-models)"]
        PKL["model_*.pkl<br/>data/artifacts/"]
    end

    subgraph Export["Exportación Métricas"]
        ExportMetrics["export_model_metrics.py"]
        ExportSHAP["export_shap_features.py<br/>(TreeExplainer XGB/GBM + LinearExplainer LR)"]
        ExportPipeline["export_pipeline_metrics.py<br/>(103 métricas — 4 etapas)"]
        ModelMetrics["debit_model_metrics"]
        ModelFeatures["debit_model_features<br/>(50 features × 3 modelos × N configs)"]
        PipelineMetrics["debit_pipeline_metrics"]
    end

    subgraph Viz["Visualización"]
        Views["v_debit_* views<br/>(9 vistas SQL)"]
        Dashboard["Metabase Dashboard<br/>id=4, :3000"]
    end

    CSV1 --> Loader
    CSV2 --> Loader
    CSV3 --> Loader
    CSV4 --> Loader
    CSV5 --> Loader
    CSV6 --> Loader
    Loader --> TClient
    Loader --> TPago
    Loader --> TGestion
    Loader --> TMora
    Loader --> TExcedente
    Loader --> TCanal

    TClient --> DataPrep
    TPago --> DataPrep
    TGestion --> DataPrep
    TMora --> DataPrep
    TExcedente --> DataPrep
    TCanal --> DataPrep
    DataPrep --> AnalyticalModel

    AnalyticalModel --> FeatEng
    FeatEng --> TrainParq
    FeatEng --> TestParq
    FeatEng --> OOTParq

    TrainParq --> FeatSel
    FeatSel --> FeatCols

    TrainParq --> TrainScript
    TestParq --> TrainScript
    OOTParq --> TrainScript
    FeatCols --> TrainScript
    TrainScript --> MLflowRuns
    TrainScript --> PKL

    MLflowRuns --> ExportMetrics
    PKL --> ExportSHAP
    ExportMetrics --> ModelMetrics
    ExportMetrics --> ModelFeatures
    ExportSHAP --> ModelFeatures
    FeatSel --> ExportPipeline
    AnalyticalModel --> ExportPipeline
    ExportPipeline --> PipelineMetrics

    TClient --> Views
    TPago --> Views
    ModelMetrics --> Dashboard
    ModelFeatures --> Dashboard
    PipelineMetrics --> Dashboard
    Views --> Dashboard
```

## Arquitectura de Entrenamiento ML

```mermaid
flowchart TB
    subgraph EntryPoint["Punto de Entrada"]
        TrainScript["train_debit_classifier.py<br/>--split-strategy --feature-set"]
    end

    subgraph Pipeline["Pipeline de Entrenamiento"]
        MainFn["main()"]
        RunFS["_run_feature_set()<br/>× Opción A | B"]
        FilterB["_filter_option_b()<br/>excluye 129 features (Opción B)"]
    end

    subgraph Dataset["Artefactos de Entrada"]
        TrainParq["train.parquet"]
        TestParq["test.parquet"]
        OOTParq["oot.parquet"]
        FeatCols["feature_cols.json"]
    end

    subgraph Models["Clasificadores — MODEL_REGISTRY"]
        XGB["XGBoostDebitClassifier<br/>scale_pos_weight + early stopping"]
        GBM["GradientBoostingDebitClassifier<br/>HistGradientBoosting"]
        LR["LogisticRegressionDebitClassifier<br/>saga solver"]
        RF["RandomForestDebitClassifier<br/>(excluido de DEFAULT_MODELS)"]
        Base["BaseDebitClassifier (ABC)<br/>+ ClassificationMetrics"]
    end

    subgraph Optimization["Optimización"]
        Optuna["Optuna<br/>n_trials=5"]
        EvalCV["evaluate_classifier_cv()<br/>StratifiedKFold"]
    end

    subgraph Tracking["Experiment Tracking"]
        MLflow["MLflow<br/>debit-models"]
        ParentRun["Parent Run por feature_set<br/>+ Child Runs por modelo"]
    end

    subgraph Scoring["Servicio de Scoring"]
        ScoringService["DebitScoringService<br/>score() → prob + segmento"]
        Segments["Segmentos A/B/C/D"]
    end

    subgraph Output["Salidas"]
        PKL["model_{name}_{A|B}.pkl<br/>data/artifacts/"]
        Metrics["AUC + KS + Precision/Recall<br/>CV / Train / Test / OOT"]
    end

    TrainScript --> MainFn
    Dataset --> MainFn

    MainFn --> RunFS
    RunFS --> FilterB
    RunFS --> XGB
    RunFS --> GBM
    RunFS --> LR

    XGB --> Base
    GBM --> Base
    LR --> Base
    RF --> Base

    RunFS --> Optuna
    Optuna --> EvalCV
    EvalCV --> Base

    RunFS --> MLflow
    MLflow --> ParentRun

    RunFS --> PKL
    RunFS --> Metrics

    PKL --> ScoringService
    ScoringService --> Segments
```

## Interacción de Servicios Docker

```mermaid
flowchart TB
    subgraph Network["debit_network"]
        subgraph Core["Servicio Base"]
            PG["postgres<br/>PostgreSQL :5432<br/>Vol: postgres-db (named)"]
        end

        subgraph AirflowStack["Stack Airflow"]
            AIInit["airflow-init<br/>debit-airflow:latest"]
            AIWeb["airflow-webserver<br/>debit-airflow:latest :8080"]
            AISched["airflow-scheduler<br/>debit-airflow:latest"]
        end

        subgraph Apps["Servicios de Aplicación"]
            ML["debit-ml<br/>debit-ml:latest"]
            MF["mlflow<br/>debit-ml:latest :5000"]
            MB["metabase<br/>metabase/metabase :3000"]
        end
    end

    subgraph Volumes["Volúmenes montados"]
        VolDatalake["datalake/"]
        VolDags["data/dags/"]
        VolArtifacts["data/artifacts/"]
        VolMLflow["data/mlflow/"]
        VolDB["postgres-db (Docker named)"]
    end

    PG --> AIInit
    AIInit --> AIWeb
    AIInit --> AISched
    PG --> ML
    PG --> MF
    PG --> MB
    MF --> ML

    VolDatalake --> ML
    VolArtifacts --> ML
    VolDags --> AIWeb
    VolDags --> AISched
    VolMLflow --> MF
    VolDB --> PG
```

## Esquema de Base de Datos

```mermaid
erDiagram
    debit_client {
        varchar num_doc PK
        varchar obl17 PK
        date f_analisis PK
        int var_rta
    }

    debit_pago {
        varchar num_doc FK
        varchar obl17 FK
        date f_analisis FK
        float avg_pago_debito_3m
        float avg_pago_debito_6m
        float avg_pago_fisico_3m
        float avg_pago_virtual_3m
        float avg_pago_otros_3m
    }

    debit_excedente {
        varchar num_doc FK
        varchar obl17 FK
        date f_analisis FK
        float avg_excedente_pago_3m
        float avg_porc_pago_3m
        float max_porc_pago_3m
    }

    debit_gestion {
        varchar num_doc FK
        varchar obl17 FK
        date f_analisis FK
        float avg_cant_gestiones_3m
        float avg_cant_rpc_3m
        float avg_cant_acuerdo_3m
        float avg_cant_promesas_3m
    }

    debit_mora {
        varchar num_doc FK
        varchar obl17 FK
        date f_analisis FK
        float moras_avg_mora_3m
        float moras_avg_mora_6m
        float moras_avg_mora_9m
    }

    debit_canal {
        varchar num_doc FK
        varchar obl17 FK
        date f_analisis FK
        float trx_mnt_total
        float trx_mnt_total_smmlv
        int trx_cnt_total
    }

    debit_model_metrics {
        varchar run_id PK
        varchar model_name
        varchar split_strategy
        varchar feature_set
        float cv_auc
        float train_auc
        float test_auc
        float oot_auc
        float train_ks
        float test_ks
        int train_size
        int test_size
        int oot_size
        timestamp run_date
    }

    debit_model_features {
        int id PK
        varchar run_id FK
        varchar model_name
        varchar feature_name
        varchar split_strategy
        varchar feature_set
        float importance
        int rank
    }

    debit_pipeline_metrics {
        int id PK
        varchar run_id
        varchar stage
        varchar metric_name
        float metric_value
        timestamp run_date
    }

    debit_client ||--|| debit_pago : "join (num_doc, obl17, f_analisis)"
    debit_client ||--|| debit_excedente : "join"
    debit_client ||--|| debit_gestion : "join"
    debit_client ||--|| debit_mora : "join"
    debit_client ||--o| debit_canal : "left join (20.7% cobertura)"
    debit_model_metrics ||--o{ debit_model_features : "run_id → 50 features SHAP"
    debit_model_metrics ||--o{ debit_pipeline_metrics : "run_id → métricas pipeline"
```

## Estructura del Proyecto

```mermaid
flowchart TB
    subgraph Root["Raíz del Proyecto"]
        subgraph Deploy["deploy/"]
            TrainScript["train_debit_classifier.py<br/>Entrypoint ML multi-modelo"]
            ExportMetrics["export_model_metrics.py<br/>MLflow → debit_model_metrics"]
            ExportSHAP["export_shap_features.py<br/>SHAP XGB/GBM/LogReg → debit_model_features"]
            ExportPipeline["export_pipeline_metrics.py<br/>pipeline → debit_pipeline_metrics"]
            MetabaseProv["metabase_provisioning.py<br/>18 cards + dashboard (idempotente)"]
            Entrypoint["entrypoint.sh<br/>alembic upgrade + cmd"]
            DF["Dockerfile"]
            DFA["Dockerfile.airflow"]
        end

        subgraph DataDir["data/"]
            DAGsDir["dags/<br/>debit_ml_pipeline_dag.py"]
            ArtifDir["artifacts/<br/>train/test/oot.parquet<br/>feature_cols.json<br/>model_*.pkl"]
            QueriesDir["queries/<br/>debit_views.sql (9 vistas)"]
        end

        subgraph Src["src/"]
            Settings["settings.py<br/>Pydantic Settings"]
            subgraph DBMod["database/"]
                Conn["connections.py<br/>engine + get_session()"]
                CRUD["crud/debit.py"]
            end
            subgraph DatasetMod["dataset/"]
                Loader["loader.py"]
                DataPrep["data_preparation.py"]
                FeatEng["feature_engineering.py"]
                FeatSel["feature_selection.py"]
                FeatConf["feature_config.py"]
                SplitStrat["split_strategies.py"]
            end
            subgraph ModelsMod["models/"]
                DebitModel["debit.py<br/>6 SQLModel tables"]
            end
            subgraph ServicesMod["services/"]
                Forecast["forecasting.py<br/>DebitScoringService"]
                Optim["optimization.py<br/>ModelSelectionService"]
            end
            subgraph StatMod["statistical_models/"]
                BaseCls["base.py + evaluation.py"]
                XGBCls["xgboost_classifier.py"]
                GBMCls["gradient_boosting_classifier.py"]
                LRCls["logistic_regression.py"]
                RFCls["random_forest_classifier.py"]
            end
        end

        subgraph Config["Configuración"]
            DC["docker-compose.yml"]
            Env[".env"]
            Pyproject["pyproject.toml"]
            MF["Makefile"]
        end
    end

    Deploy --> Src
    DAGsDir --> Deploy
    Src --> DBMod
    StatMod --> BaseCls
```

## Dependencias del DAG — debit_ml_pipeline

```mermaid
flowchart LR
    subgraph Gates["ShortCircuitOperators (gates)"]
        G0["gate_setup<br/>RUN_SETUP"]
        G1["gate_load_data<br/>RUN_LOAD_DATA"]
        G2["gate_build_model<br/>RUN_BUILD_MODEL"]
        G3["gate_build_features<br/>RUN_BUILD_FEATURES"]
        G35["gate_select_features<br/>RUN_SELECT_FEATURES"]
        G4["gate_train<br/>RUN_TRAIN"]
        G5["gate_metabase<br/>RUN_METABASE"]
    end

    subgraph Tasks["DockerOperator Tasks (debit-ml:latest)"]
        T0["T0 setup_artifacts_dir<br/>mkdir /app/data/artifacts"]
        T1["T1 load_data<br/>src.dataset.loader"]
        T2["T2 build_analytical_model<br/>src.dataset.data_preparation"]
        T3["T3 build_features<br/>src.dataset.feature_engineering"]
        T35["T3.5 select_features<br/>src.dataset.feature_selection"]
        T4["T4 train_model<br/>deploy.train_debit_classifier"]
        T5["T5 provision_metabase<br/>deploy.metabase_provisioning"]
    end

    G0 --> T0
    T0 --> G1
    G1 --> T1
    T1 --> G2
    G2 --> T2
    T2 --> G3
    G3 --> T3
    T3 --> G35
    G35 --> T35
    T35 --> G4
    G4 --> T4
    T4 --> G5
    G5 --> T5
```

## Segmentación de Cobranza

```mermaid
flowchart TB
    subgraph Input["Entrada — DebitScoringService"]
        PKL["model_*.pkl<br/>(gradient_boosting — mejor AUC OOT)"]
        Features["60 features (Opción A) / 19 features (Opción B)<br/>feature_cols.json"]
    end

    subgraph Score["Scoring"]
        ScoreFn["score() → probabilidad clase 1"]
    end

    subgraph Segments["Segmentos de Cobranza"]
        SegA["Segmento A — Automatizar<br/>var_rta=1 y mora_6m ≤ 10 días"]
        SegB["Segmento B — Monitorear<br/>var_rta=1 y mora_6m > 10 días"]
        SegC["Segmento C — Cobranza suave<br/>var_rta=0 y mora_6m ≤ 15 días"]
        SegD["Segmento D — Cobranza intensiva<br/>var_rta=0 y mora_6m > 15 días"]
    end

    subgraph Views["Vistas Metabase"]
        VRisk["v_debit_risk_segment_summary"]
        VKPI["v_debit_kpi_period"]
        VMix["v_debit_payment_mix"]
        VGest["v_debit_gestiones_profile"]
    end

    PKL --> ScoreFn
    Features --> ScoreFn
    ScoreFn --> SegA
    ScoreFn --> SegB
    ScoreFn --> SegC
    ScoreFn --> SegD

    SegA --> VRisk
    SegB --> VRisk
    SegC --> VRisk
    SegD --> VRisk
    VRisk --> VKPI
    VRisk --> VMix
    VRisk --> VGest
```

## Stack Tecnológico

| Componente | Tecnología | Propósito |
|------------|------------|-----------|
| Orquestación | Apache Airflow 2.8.1 | Scheduling y ejecución del DAG |
| Base de datos | PostgreSQL 17 | Persistencia de datos y vistas Metabase |
| ML Tracking | MLflow | Registro de experimentos y artefactos |
| Optimización | Optuna | Hyperparameter tuning (n_trials=5) |
| Visualización | Metabase | Dashboard de cobranza (18 cards) |
| Contenedores | Docker Compose | Orquestación de servicios |
| Gestor de paquetes | UV | Gestión de dependencias Python |
| Modelos ML | XGBoost, HistGBM, LogisticRegression | Clasificación binaria var_rta |
| Interpretabilidad | SHAP (TreeExplainer + LinearExplainer) | Feature importance para XGBoost, HistGBM y LogisticRegression |
| ORM | SQLModel + SQLAlchemy | Modelos de datos y migraciones |
| Migraciones | Alembic | Schema versioning — 5 migraciones aplicadas |
| Parametrización | `split_strategy` × `feature_set` | 4 configuraciones: A/B × random/temporal |

---

## Hallazgos Críticos del EDA

```mermaid
flowchart TB
    subgraph H1["H1 — Hallazgo más importante"]
        direction LR
        C0["Clase 0 (9,901 obs)"]
        C0Pay["total_pago = 0\nen TODOS los canales\ny TODAS las ventanas"]
        C0 --> C0Pay
    end

    subgraph Implication1["Implicación H1"]
        I1A["var_rta=0 ≠ 'pagó por otro canal'"]
        I1B["var_rta=0 = 'sin actividad de pago'"]
        I1A --> I1B
    end

    subgraph H2["H2 — Ventana temporal"]
        V1["14.9% clase-1 con pagos físico/virtual en 3m"]
        V2["var_rta calculado en f_analisis puntual\ntanque promedia los 3m previos"]
        V1 --> V2
    end

    subgraph H4["H4 — Estructura del dataset"]
        T1["97.9% obligaciones\nen único f_analisis"]
        T2["Dataset casi transversal\nno longitudinal"]
        T3["Split temporal segmenta\ncohortes distintas"]
        T1 --> T2 --> T3
    end

    subgraph Diagnosis["Diagnóstico AUC = 1.0"]
        D1["H1 + H4 interactúan"]
        D2["Split temporal concentra\nclase-0 en Test/OOT"]
        D3["Features débito discriminan\ntrivialmente → AUC = 1.0"]
        D4["Split aleatorio → AUC = 0.96\n(valor real del modelo)"]
        D1 --> D2 --> D3 --> D4
    end

    H1 --> Implication1
    Implication1 --> Diagnosis
    H4 --> Diagnosis
```

---

## Resultados del Modelo — Métricas Clave

```mermaid
flowchart LR
    subgraph OptionA["Opción A — Producción (60 features)"]
        direction TB
        GBM_A["Gradient Boosting ⭐\nCV AUC: 0.9616\nTest AUC: 0.9620\nOOT AUC: 0.9616\nKS Test: 0.7379"]
        XGB_A["XGBoost\nCV AUC: 0.9587\nTest AUC: 0.9594\nOOT AUC: 0.9580\nKS Test: 0.7300"]
        LR_A["Logistic Regression\nCV AUC: 0.9548\nTest AUC: 0.9537\nOOT AUC: 0.9530\nKS Test: 0.7100"]
    end

    subgraph OptionB["Opción B — Sin historial (19 features)"]
        direction TB
        GBM_B["Gradient Boosting\nTest AUC: ~0.8875"]
        XGB_B["XGBoost\nTest AUC: ~0.87"]
        LR_B["Logistic Regression\nTest AUC: ~0.84"]
    end

    subgraph Stability["Estabilidad — sin sobreajuste"]
        ST["Train ≈ Test ≈ OOT\nen todas las configuraciones"]
    end

    subgraph TopFeature["Top Feature (SHAP)"]
        TF["prop_debito_3m\nimportance = 8.68\n(proporción de pagos\npor débito en 3 meses)"]
    end

    GBM_A --> Stability
    GBM_B --> Stability
    Stability --> TopFeature
```

---

## Flujo de Valor — Impacto en Cobranza

```mermaid
flowchart TB
    subgraph Input["Entrada — Cartera en Mora Temprana (1–30 días)"]
        Cartera["45,731 obligaciones\n(universo analizado)"]
    end

    subgraph Model["Modelo de Clasificación\nGradient Boosting · AUC = 0.96"]
        Scoring["DebitScoringService\nscore() → P(var_rta=1)"]
    end

    subgraph Segments["Segmentación Operativa"]
        SegA["SEGMENTO A\n~35% de la cartera\nvar_rta=1 · mora ≤ 10d\n→ AUTOMATIZAR\nSin gestión humana"]
        SegB["SEGMENTO B\n~44% de la cartera\nvar_rta=1 · mora > 10d\n→ MONITOREAR\nSeguimiento ligero"]
        SegC["SEGMENTO C\n~10% de la cartera\nvar_rta=0 · mora ≤ 15d\n→ COBRANZA SUAVE\nContacto preventivo"]
        SegD["SEGMENTO D\n~11% de la cartera\nvar_rta=0 · mora > 15d\n→ COBRANZA INTENSIVA\nGestión activa"]
    end

    subgraph Impact["Impacto Operativo"]
        Auto["~79% del volumen\nsin costo de gestión"]
        Focus["Gestión humana\nfocalizada en C+D\n(~21% del volumen)"]
    end

    Cartera --> Scoring
    Scoring --> SegA
    Scoring --> SegB
    Scoring --> SegC
    Scoring --> SegD
    SegA --> Auto
    SegB --> Auto
    SegC --> Focus
    SegD --> Focus
```

---

## Conclusiones y Accionables de Negocio

```mermaid
flowchart TB
    subgraph Findings["Conclusiones Técnicas"]
        F1["✅ Modelo válido y robusto\nAUC=0.96 · KS=0.74\nTrain≈Test≈OOT"]
        F2["✅ H1 — hallazgo crítico\nClase-0 = sin actividad de pago\nRedefine objetivo del modelo"]
        F3["✅ Selección supervisada confirma EDA\npago_fisico y pago_otros\neliminados automáticamente por ANOVA"]
        F4["✅ Diagnóstico transparente\nAUC=1.0 temporal = causa estructural\nno sobreajuste ni leakage"]
        F5["✅ Opción B agrega valor\nAUC=0.88 solo con gestiones+moras\nComportamiento de cobranza predice el pago"]
    end

    subgraph Actions["Accionables Inmediatos"]
        A1["🔴 Alta prioridad\nActivar Segmento A\nDesactivar gestión humana"]
        A2["🔴 Alta prioridad\nAjustar umbral operativo\nsegún costo-beneficio real"]
        A3["🟡 Media prioridad\nDesplegar Opción B\npara clientes sin historial"]
        A4["🟡 Media prioridad\nMonitoreo automático\nPSI mensual sobre prop_debito_3m"]
        A5["🟢 Normal\nRe-entrenamiento automático\nsi AUC OOT < 0.90"]
    end

    subgraph Requirements["Requerimientos Cumplidos"]
        R1["R1 ✅ Modelo de datos analítico\n(num_doc, obl17, f_analisis)"]
        R2["R2 ✅ Pipeline escalable\nAirflow DAG @monthly"]
        R3["R3 ✅ Split Train/Test/OOT\nJustificado + diagnóstico resuelto"]
        R4["R4 ✅ AUC=0.96 + KS + P/R/F1\n+ Brier + Calibración + SHAP"]
        R5["R5 ✅ Dashboard Metabase\n18 cards · accionables de negocio"]
    end

    Findings --> Actions
    Actions --> Requirements
```
