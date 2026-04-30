 Reporte de Reconocimiento del Proyecto — Débitos Recurrentes Bancolombia                                                                                                                                                                                                    
  1. Árbol de Carpetas del Proyecto                                                                                                                                                                                                                                           
  approach2/                                                                                                                            
  ├── CLAUDE.md                          # Instrucciones del proyecto
  ├── README.md
  ├── docker-compose.yml                 # Infraestructura: Airflow, MLflow, Metabase, Postgres
  ├── makefile
  ├── pyproject.toml
  ├── alembic.ini                        # Migraciones de BD
  ├── alembic/
  │   ├── env.py
  │   └── versions/                      # Migraciones generadas
  ├── data/
  │   ├── dags/                          # DAGs de Apache Airflow
  │   │   ├── coin_backfill_dag.py
  │   │   ├── coin_fetcher_dag.py
  │   │   ├── coin_statistical_analysis_dag.py
  │   │   ├── coin_train_forecast.py
  │   │   └── coin_train_forecast_pct.py
  │   ├── sql_queries/
  │   │   ├── coin_analysis.sql
  │   │   └── sp_analyze_coin_statistical.sql
  │   ├── mlflow/artifacts/              # Artefactos de MLflow
  │   └── database/                      # Datos binarios PostgreSQL (volumen Docker)
  ├── datalake/                          # ← FUENTE DE DATOS DEL EXAMEN
  │   ├── clientes_seleccionados_prueba.csv
  │   ├── debitos_dev_cruce_canales_poblacion_sel_vars_prueba_tecnica_1.csv
  │   ├── debitos_dev_cruce_excedente_poblacion_prueba_tecnica.csv
  │   ├── debitos_dev_cruce_gestiones_poblacion_prueba_tecnica.csv
  │   ├── debitos_dev_cruce_moras_poblacion_prueba_tecnica.csv
  │   └── debitos_dev_cruce_tanque_poblacion_prueba_tecnica.csv
  ├── deploy/
  │   ├── coin_fetcher.py
  │   ├── Dockerfile
  │   ├── entrypoint.sh
  │   └── train_forecast_models_pct.py
  ├── docker/
  │   └── postgres/
  │       └── init-databases.sh
  ├── figures/                           # Visualizaciones generadas
  ├── notebooks/
  │   ├── statistical_analysis.ipynb
  │   └── statistical_analysis_of_time_series.ipynb
  ├── scripts/
  │   └── export_pdf.py
  ├── src/
  │   ├── database/
  │   │   ├── connections.py
  │   │   └── crud/coin.py
  │   ├── dataset/
  │   │   ├── data_preparation.py
  │   │   ├── feature_config.py
  │   │   └── feature_engineering.py
  │   ├── eda/
  │   │   ├── lag_features.py
  │   │   ├── modeling.py
  │   │   ├── risk_classification.py
  │   │   ├── time_features.py
  │   │   ├── visualization.py
  │   │   └── volume_features.py
  │   ├── models/coin.py
  │   ├── services/
  │   │   ├── forecasting.py
  │   │   ├── optimization.py
  │   │   └── training.py
  │   ├── statistical_models/
  │   │   ├── base.py, elasticnet.py, evaluation.py
  │   │   ├── gradient_boosting.py, prophet_model.py
  │   │   ├── random_forest.py, ridge.py, sarimax.py
  │   └── settings.py
  └── templates/custom_report/           # Plantillas de exportación PDF

  ---
  2. Análisis por Archivo CSV

  2.1 clientes_seleccionados_prueba.csv — Tabla maestra / variable respuesta

  ┌────────────────────────────────┬──────────────────────────┐
  │            Atributo            │          Valor           │
  ├────────────────────────────────┼──────────────────────────┤
  │ Filas                          │ 46,736                   │
  ├────────────────────────────────┼──────────────────────────┤
  │ Columnas                       │ 4                        │
  ├────────────────────────────────┼──────────────────────────┤
  │ Claves únicas (num_doc, obl17) │ 45,731                   │
  ├────────────────────────────────┼──────────────────────────┤
  │ Periodos únicos f_analisis     │ 17 (Jul 2024 – Nov 2025) │
  ├────────────────────────────────┼──────────────────────────┤
  │ Valores nulos                  │ Ninguno                  │
  └────────────────────────────────┴──────────────────────────┘

  ┌────────────┬───────────────┬──────────────────────────────────────────────────────────────────┐
  │  Columna   │ Tipo inferido │                           Descripción                            │
  ├────────────┼───────────────┼──────────────────────────────────────────────────────────────────┤
  │ num_doc    │ object (hash) │ Identificador anonimizado del cliente                            │
  ├────────────┼───────────────┼──────────────────────────────────────────────────────────────────┤
  │ obl17      │ object (hash) │ Identificador anonimizado de la obligación                       │
  ├────────────┼───────────────┼──────────────────────────────────────────────────────────────────┤
  │ f_analisis │ object → date │ Fecha de corte del análisis (mensual)                            │
  ├────────────┼───────────────┼──────────────────────────────────────────────────────────────────┤
  │ var_rta    │ int64         │ Variable objetivo: 1 = débito recurrente ≥40%, 0 = otros canales │
  └────────────┴───────────────┴──────────────────────────────────────────────────────────────────┘

  Distribución de var_rta:

  ┌───────────────────────┬────────┬────────────┐
  │         Clase         │ Conteo │ Proporción │
  ├───────────────────────┼────────┼────────────┤
  │ 1 (débito recurrente) │ 36,835 │ 78.8%      │
  ├───────────────────────┼────────┼────────────┤
  │ 0 (otros canales)     │ 9,901  │ 21.2%      │
  └───────────────────────┴────────┴────────────┘

  ▎ ⚠️  Dataset desbalanceado (ratio 3.7:1 a favor de la clase positiva).

  ---
  2.2 debitos_dev_cruce_canales_..._1.csv — Transacciones por canal

  ┌────────────────────────────────┬──────────────────────────────────────────────────────────────┐
  │            Atributo            │                            Valor                             │
  ├────────────────────────────────┼──────────────────────────────────────────────────────────────┤
  │ Filas                          │ 10,801                                                       │
  ├────────────────────────────────┼──────────────────────────────────────────────────────────────┤
  │ Columnas                       │ 4,258                                                        │
  ├────────────────────────────────┼──────────────────────────────────────────────────────────────┤
  │ Claves únicas (num_doc, obl17) │ 9,460                                                        │
  ├────────────────────────────────┼──────────────────────────────────────────────────────────────┤
  │ Periodos únicos f_analisis     │ 17                                                           │
  ├────────────────────────────────┼──────────────────────────────────────────────────────────────┤
  │ Valores nulos                  │ 3.45% en todas las variables (373 filas completas sin datos) │
  └────────────────────────────────┴──────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────┬─────────┬────────────────────────────────────────────────────┐
  │              Columna               │  Tipo   │                    Descripción                     │
  ├────────────────────────────────────┼─────────┼────────────────────────────────────────────────────┤
  │ num_doc, obl17, f_analisis         │ object  │ Llaves de join                                     │
  ├────────────────────────────────────┼─────────┼────────────────────────────────────────────────────┤
  │ trx_mnt_total, trx_mnt_total_smmlv │ float64 │ Monto total de transacciones (absoluto y en SMMLV) │
  ├────────────────────────────────────┼─────────┼────────────────────────────────────────────────────┤
  │ trx_cnt_total                      │ float64 │ Conteo total de transacciones                      │
  ├────────────────────────────────────┼─────────┼────────────────────────────────────────────────────┤
  │ trx_cnt_[canal]_[tipo]             │ float64 │ Conteo por canal y tipo de transacción             │
  ├────────────────────────────────────┼─────────┼────────────────────────────────────────────────────┤
  │ trx_mnt_[canal]_[tipo]             │ float64 │ Monto por canal y tipo de transacción              │
  ├────────────────────────────────────┼─────────┼────────────────────────────────────────────────────┤
  │ trx_cnt_can_[canal]                │ float64 │ Conteo de canales distintos usados                 │
  ├────────────────────────────────────┼─────────┼────────────────────────────────────────────────────┤
  │ trx_cnt_gr_[tipo]                  │ float64 │ Conteo agrupado por tipo de operación              │
  └────────────────────────────────────┴─────────┴────────────────────────────────────────────────────┘

  Canales identificados: app_per, app_pyme, bill_mvl, btn_bco, cajero, cr_bcrio, pos, pse_emp, pse_per, rec_e_v, suc_fis, suc_tel,
  sv_pyme, sve, svp

  Tipos de operación: pag (pago), pag_ob (pago obligación), trf (transferencia), tfy, ava (avance), ret (retiro), cmpr (compra), dep
  (depósito), dvs, rcg, ut_c_cr

  ▎ ⚠️  Este archivo cubre solo 9,460 de 45,731 obligaciones (20.7%) — subconjunto crítico.

  ---
  2.3 debitos_dev_cruce_excedente_poblacion_prueba_tecnica.csv — Excedentes de pago

  ┌────────────────────────────────┬────────┐
  │            Atributo            │ Valor  │
  ├────────────────────────────────┼────────┤
  │ Filas                          │ 47,194 │
  ├────────────────────────────────┼────────┤
  │ Columnas                       │ 35     │
  ├────────────────────────────────┼────────┤
  │ Claves únicas (num_doc, obl17) │ 45,731 │
  ├────────────────────────────────┼────────┤
  │ Periodos únicos f_analisis     │ 17     │
  └────────────────────────────────┴────────┘

  ┌────────────────────────────────────────┬─────────┬─────────────────┐
  │           Grupo de columnas            │  Tipo   │    Ventanas     │
  ├────────────────────────────────────────┼─────────┼─────────────────┤
  │ avg/min/max/stddev_excedente_pago_[Xm] │ float64 │ 3m, 6m, 9m, 12m │
  ├────────────────────────────────────────┼─────────┼─────────────────┤
  │ avg/min/max/stddev_porc_pago_[Xm]      │ float64 │ 3m, 6m, 9m, 12m │
  └────────────────────────────────────────┴─────────┴─────────────────┘

  Valores nulos por ventana temporal:

  ┌─────────────────┬─────────────┬───────┐
  │    Variable     │    Nulos    │   %   │
  ├─────────────────┼─────────────┼───────┤
  │ *_porc_pago_3m  │ 480         │ 1.02% │
  ├─────────────────┼─────────────┼───────┤
  │ *_porc_pago_6m  │ 1,081       │ 2.29% │
  ├─────────────────┼─────────────┼───────┤
  │ *_porc_pago_9m  │ 1,378       │ 2.92% │
  ├─────────────────┼─────────────┼───────┤
  │ *_porc_pago_12m │ ~1,600 est. │ ~3.4% │
  └─────────────────┴─────────────┴───────┘

  ▎ Patrón esperado: nulos crecen con ventanas más largas (obligaciones nuevas sin historial completo).

  ---
  2.4 debitos_dev_cruce_gestiones_poblacion_prueba_tecnica.csv — Gestiones de cobranza

  ┌────────────────────────────────┬─────────┐
  │            Atributo            │  Valor  │
  ├────────────────────────────────┼─────────┤
  │ Filas                          │ 47,194  │
  ├────────────────────────────────┼─────────┤
  │ Columnas                       │ 83      │
  ├────────────────────────────────┼─────────┤
  │ Claves únicas (num_doc, obl17) │ 45,731  │
  ├────────────────────────────────┼─────────┤
  │ Periodos únicos f_analisis     │ 17      │
  ├────────────────────────────────┼─────────┤
  │ Valores nulos                  │ Ninguno │
  └────────────────────────────────┴─────────┘

  ┌────────────────────────────────────────────┬───────────────┬─────────────────┐
  │             Grupo de columnas              │     Tipo      │    Ventanas     │
  ├────────────────────────────────────────────┼───────────────┼─────────────────┤
  │ avg/min/max/stddev_cant_gestiones_[Xm]     │ float64/int64 │ 3m, 6m, 9m, 12m │
  ├────────────────────────────────────────────┼───────────────┼─────────────────┤
  │ avg/min/max/stddev_cant_rpc_[Xm]           │ float64       │ 3m, 6m, 9m, 12m │
  ├────────────────────────────────────────────┼───────────────┼─────────────────┤
  │ avg/min/max/stddev_cant_acuerdo_[Xm]       │ float64       │ 3m, 6m, 9m, 12m │
  ├────────────────────────────────────────────┼───────────────┼─────────────────┤
  │ avg/min/max/stddev_promesas_cumplidas_[Xm] │ float64       │ 3m, 6m, 9m, 12m │
  ├────────────────────────────────────────────┼───────────────┼─────────────────┤
  │ avg/min/max/stddev_maximo_rank_[Xm]        │ float64       │ 3m, 6m, 9m, 12m │
  └────────────────────────────────────────────┴───────────────┴─────────────────┘

  ▎ rpc = Respuesta Por Cliente, acuerdo = acuerdos de pago, rank = nivel de intensidad de gestión.

  ---
  2.5 debitos_dev_cruce_moras_poblacion_prueba_tecnica.csv — Mora por obligación

  ┌────────────────────────────────┬─────────┐
  │            Atributo            │  Valor  │
  ├────────────────────────────────┼─────────┤
  │ Filas                          │ 47,194  │
  ├────────────────────────────────┼─────────┤
  │ Columnas                       │ 12      │
  ├────────────────────────────────┼─────────┤
  │ Claves únicas (num_doc, obl17) │ 45,731  │
  ├────────────────────────────────┼─────────┤
  │ Periodos únicos f_analisis     │ 17      │
  ├────────────────────────────────┼─────────┤
  │ Valores nulos                  │ Ninguno │
  └────────────────────────────────┴─────────┘

  ┌────────────────────────────────────┬───────────────┬──────────────────────┐
  │              Columna               │     Tipo      │ Ventanas disponibles │
  ├────────────────────────────────────┼───────────────┼──────────────────────┤
  │ moras_avg/min/max/stddev_mora_[Xm] │ float64/int64 │ 3m, 6m, 9m (sin 12m) │
  └────────────────────────────────────┴───────────────┴──────────────────────┘

  ▎ ⚠️  Anomalía: este archivo solo tiene ventanas hasta 9m — falta la ventana de 12m presente en los demás archivos.

  ---
  2.6 debitos_dev_cruce_tanque_poblacion_prueba_tecnica.csv — Pagos por tipo de canal

  ┌────────────────────────────────┬─────────┐
  │            Atributo            │  Valor  │
  ├────────────────────────────────┼─────────┤
  │ Filas                          │ 47,194  │
  ├────────────────────────────────┼─────────┤
  │ Columnas                       │ 67      │
  ├────────────────────────────────┼─────────┤
  │ Claves únicas (num_doc, obl17) │ 45,731  │
  ├────────────────────────────────┼─────────┤
  │ Periodos únicos f_analisis     │ 17      │
  ├────────────────────────────────┼─────────┤
  │ Valores nulos                  │ Ninguno │
  └────────────────────────────────┴─────────┘

  ┌──────────────────────────────────────┬─────────┬─────────────────┐
  │          Grupo de columnas           │  Tipo   │    Ventanas     │
  ├──────────────────────────────────────┼─────────┼─────────────────┤
  │ avg/min/max/stddev_pago_debito_[Xm]  │ float64 │ 3m, 6m, 9m, 12m │
  ├──────────────────────────────────────┼─────────┼─────────────────┤
  │ avg/min/max/stddev_pago_fisico_[Xm]  │ float64 │ 3m, 6m, 9m, 12m │
  ├──────────────────────────────────────┼─────────┼─────────────────┤
  │ avg/min/max/stddev_pago_virtual_[Xm] │ float64 │ 3m, 6m, 9m, 12m │
  ├──────────────────────────────────────┼─────────┼─────────────────┤
  │ avg/min/max/stddev_pago_otros_[Xm]   │ float64 │ 3m, 6m, 9m, 12m │
  └──────────────────────────────────────┴─────────┴─────────────────┘

  ▎ Este archivo contiene directamente la señal de débito (variable más relevante para el modelo).

  ---
  3. Identificadores Comunes — Claves de Join

  Clave compuesta principal

  (num_doc, obl17, f_analisis)  →  granularidad: cliente × obligación × periodo

  Cobertura de obligaciones por archivo

  ┌────────────────┬────────┬─────────────────────────┬───────────────────────┐
  │    Archivo     │ Filas  │ Únicos (num_doc, obl17) │ Cobertura vs clientes │
  ├────────────────┼────────┼─────────────────────────┼───────────────────────┤
  │ clientes       │ 46,736 │ 45,731                  │ 100% (base)           │
  ├────────────────┼────────┼─────────────────────────┼───────────────────────┤
  │ excedentes     │ 47,194 │ 45,731                  │ 100%                  │
  ├────────────────┼────────┼─────────────────────────┼───────────────────────┤
  │ gestiones      │ 47,194 │ 45,731                  │ 100%                  │
  ├────────────────┼────────┼─────────────────────────┼───────────────────────┤
  │ moras          │ 47,194 │ 45,731                  │ 100%                  │
  ├────────────────┼────────┼─────────────────────────┼───────────────────────┤
  │ pagos (tanque) │ 47,194 │ 45,731                  │ 100%                  │
  ├────────────────┼────────┼─────────────────────────┼───────────────────────┤
  │ canales        │ 10,801 │ 9,460                   │ 20.7% ⚠️               │
  └────────────────┴────────┴─────────────────────────┴───────────────────────┘

  Estrategia de join recomendada

  # Tabla base: clientes (con var_rta)
  # LEFT JOIN con cada tabla de features por (num_doc, obl17, f_analisis)
  # canales se une con LEFT JOIN → ~79.3% tendrá NaN → imputar como 0

  ---
  4. Supuestos Iniciales Sugeridos

  Supuesto 1 — Granularidad y unicidad

  La llave primaria del modelo analítico es (num_doc, obl17, f_analisis). Un cliente puede tener múltiples obligaciones; cada obligación   tiene un registro por periodo mensual. Los 46,736 registros en clientes vs 45,731 únicos (num_doc, obl17) sugieren que algunas
  obligaciones aparecen en múltiples cortes (lo cual es esperado en la serie mensual de 17 periodos).

  Supuesto 2 — Anonimización

  num_doc y obl17 son identificadores hasheados. No se puede inferir tipo de documento ni producto bancario directamente desde el
  identificador. Se usan solo como llaves de join.

  Supuesto 3 — var_rta es estable por obligación

  Se asume que la variable respuesta no varía significativamente entre periodos para la misma obligación (es una característica del
  comportamiento del cliente). Si cambia entre periodos, el periodo más reciente o más relevante se usa para el label.

  Supuesto 4 — Canales es un subconjunto con actividad transaccional

  Las 9,460 obligaciones en canales son aquellas con actividad transaccional registrada en el periodo. El 79.3% restante con nulos en
  canales se interpreta como sin transacciones (imputar con 0, no eliminar registros).

  Supuesto 5 — Nulos en excedentes por historial insuficiente

  Los nulos crecientes con la ventana temporal (1% a 3m, ~3.4% a 12m) corresponden a obligaciones jóvenes sin suficiente historial. Se
  imputarán con 0 o con la media sectorial, según el caso.

  Supuesto 6 — Moras: ventana 12m ausente intencionalmente

  La ausencia de la variable moras_*_mora_12m puede ser intencional (no disponible en producción) o un error de extracción. Se tratará
  como feature faltante hasta confirmar con la fuente.

  Supuesto 7 — Esquema temporal Train/Test/OOT

  Con 17 periodos (Jul 2024 – Nov 2025) se propone:
  - Train: Jul 2024 – Feb 2025 (8 meses)
  - Test (in-sample): Mar 2025 – Jul 2025 (5 meses)
  - OOT (out-of-time): Ago 2025 – Nov 2025 (4 meses)

  Supuesto 8 — Desbalance de clases requiere tratamiento

  La proporción 78.8% / 21.2% requiere estratificación en los splits, uso de class_weight='balanced' o técnicas de sobremuestreo
  (SMOTE). La métrica principal AUC es adecuada por ser insensible al umbral, pero se complementará con Precision/Recall y KS statistic
  (estándar en cobranza).

  Supuesto 9 — pagos (tanque) es la fuente más directa del target

  Las columnas avg_pago_debito_* en el archivo tanque representan directamente la variable que define var_rta. Se debe tener precaución
  con data leakage: si estas variables se calculan en el mismo periodo que se etiqueta el target, pueden ser demasiado predictivas por
  construcción.

  Supuesto 10 — Canales (4,258 cols) requiere selección de features

  El archivo canales tiene una dimensionalidad extrema (~4,255 features después de quitar las llaves). Se aplicará selección de features   (importancia por árbol, correlación, eliminación de varianza cero) antes del modelado para evitar maldición de la dimensionalidad y
  Las columnas avg_pago_debito_* en el archivo tanque representan directamente la variable que define var_rta. Se debe tener precaución
  con data leakage: si estas variables se calculan en el mismo periodo que se etiqueta el target, pueden ser demasiado predictivas por
  construcción.

  Supuesto 10 — Canales (4,258 cols) requiere selección de features

  El archivo canales tiene una dimensionalidad extrema (~4,255 features después de quitar las llaves). Se aplicará selección de features   (importancia por árbol, correlación, eliminación de varianza cero) antes del modelado para evitar maldición de la dimensionalidad y
  overfitting.

  ---
  Tengo una imagen clara del terreno. Los próximos pasos naturales son: (1) construir el modelo de datos analítico unificado con el join   correcto, (2) análisis exploratorio de las features más relevantes (pagos y canales), y (3) definir la estrategia de split temporal.
  ¿Por cuál quieres continuar?
  ---
  Tengo una imagen clara del terreno. Los próximos pasos naturales son: (1) construir el modelo de datos analítico unificado con el join   correcto, (2) análisis exploratorio de las features más relevantes (pagos y canales), y (3) definir la estrategia de split temporal.
  ¿Por cuál quieres continuar?

✻ Sautéed for 7m 0s

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Supuesto 10 — Canales (4,258 cols) requiere selección de features

  El archivo canales tiene una dimensionalidad extrema (~4,255 features después de quitar las llaves). Se aplicará selección de features   (importancia por árbol, correlación, eliminación de varianza cero) antes del modelado para evitar maldición de la dimensionalidad y
  overfitting.

  ---