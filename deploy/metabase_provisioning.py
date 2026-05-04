"""Provisionamiento automático del tablero Metabase para débitos recurrentes.

Configura Metabase vía API REST al iniciar el contenedor:
  1. Espera a que Metabase esté disponible (health check).
  2. Obtiene (o renueva) sesión de administrador.
  3. Crea la conexión a debitdb si no existe.
  4. Crea las preguntas (questions/cards) del tablero de negocio.
  5. Ensambla las preguntas en un dashboard llamado 'Débitos Recurrentes'.

Ejecución standalone (fuera de Docker):
    METABASE_URL=http://localhost:3000 \
    METABASE_USER=admin@bancolombia.com \
    METABASE_PASS=Debit2026!Bancolombia \
    DEBIT_DB_HOST=localhost \
    DEBIT_DB_NAME=debitdb \
    DEBIT_DB_USER=fredy \
    DEBIT_DB_PASS=password \
    uv run python deploy/metabase_provisioning.py

En Docker (llamado desde DAG — las credenciales se inyectan vía CONTAINER_ENV):
    python /app/deploy/metabase_provisioning.py

Usar `make provision-metabase` (desde el host) o `make airflow-trigger` (vía DAG).
"""

import logging
import os
import time

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── Configuración ─────────────────────────────────────────────────────────────

METABASE_URL  = os.getenv("METABASE_URL",  "http://metabase:3000")
MB_USER       = os.getenv("METABASE_USER", "admin@bancolombia.com")
MB_PASS       = os.getenv("METABASE_PASS", "Debit2026!Bancolombia")

DB_HOST       = os.getenv("DEBIT_DB_HOST", "postgres")
DB_NAME       = os.getenv("DEBIT_DB_NAME") or os.getenv("POSTGRES_DB", "debitdb")
DB_USER       = os.getenv("DEBIT_DB_USER") or os.getenv("POSTGRES_USER", "fredy")
DB_PASS       = os.getenv("DEBIT_DB_PASS") or os.getenv("POSTGRES_PASSWORD", "password")
DB_PORT       = int(os.getenv("DEBIT_DB_PORT", "5432"))

DASHBOARD_NAME = "Débitos Recurrentes — Bancolombia"
DB_DISPLAY     = "debitdb (Débitos)"

HEALTH_RETRIES = 20
HEALTH_WAIT    = 15  # segundos


# ─── Definición de preguntas del tablero ──────────────────────────────────────

# Cada entrada: (nombre, query SQL, tipo_visualización[, viz_settings])
# viz_settings es opcional (dict); se pasa como visualization_settings en Metabase.
_QUESTIONS: list[tuple] = [
    (
        "1. Evolución % Débito Exclusivo por Período",
        """
        SELECT periodo, pct_clase1, pct_clase0, total_obligaciones
        FROM v_debit_kpi_period
        ORDER BY periodo
        """,
        "line",
    ),
    (
        "2. Volumen de Obligaciones por Período",
        """
        SELECT periodo, total_obligaciones, obligaciones_clase1, obligaciones_clase0
        FROM v_debit_kpi_period
        ORDER BY periodo
        """,
        "bar",
    ),
    (
        "3. Segmentos de Cobranza — Distribución Global",
        """
        SELECT segmento_cobranza,
               SUM(n_obligaciones)          AS total_obligaciones,
               ROUND(AVG(pct_del_total), 2) AS pct_promedio,
               ROUND(AVG(mora_promedio_6m), 2) AS mora_promedio
        FROM v_debit_risk_segment_summary
        GROUP BY segmento_cobranza
        ORDER BY segmento_cobranza
        """,
        "pie",
    ),
    (
        "4. Mix de Canales de Pago por Clase",
        """
        SELECT clase::text,
               canal,
               ROUND(avg_pago::numeric, 4) AS avg_pago_6m
        FROM (
            SELECT clase, 'debito'  AS canal, AVG(avg_debito_6m)  AS avg_pago FROM v_debit_payment_mix GROUP BY clase
            UNION ALL
            SELECT clase, 'fisico'  AS canal, AVG(avg_fisico_6m)  AS avg_pago FROM v_debit_payment_mix GROUP BY clase
            UNION ALL
            SELECT clase, 'virtual' AS canal, AVG(avg_virtual_6m) AS avg_pago FROM v_debit_payment_mix GROUP BY clase
            UNION ALL
            SELECT clase, 'otros'   AS canal, AVG(avg_otros_6m)   AS avg_pago FROM v_debit_payment_mix GROUP BY clase
        ) t
        ORDER BY clase, canal
        """,
        "bar",
    ),
    (
        "5. Efectividad de Gestiones de Cobranza por Clase",
        """
        SELECT clase::text,
               metrica,
               ROUND(valor::numeric, 3) AS valor_promedio
        FROM (
            SELECT clase, 'gestiones'         AS metrica, AVG(avg_gestiones_6m)         AS valor FROM v_debit_gestiones_profile GROUP BY clase
            UNION ALL
            SELECT clase, 'rpc'               AS metrica, AVG(avg_rpc_6m)               AS valor FROM v_debit_gestiones_profile GROUP BY clase
            UNION ALL
            SELECT clase, 'acuerdos'          AS metrica, AVG(avg_acuerdos_6m)          AS valor FROM v_debit_gestiones_profile GROUP BY clase
            UNION ALL
            SELECT clase, 'promesas_cumplidas' AS metrica, AVG(avg_promesas_cumplidas_6m) AS valor FROM v_debit_gestiones_profile GROUP BY clase
        ) t
        ORDER BY clase, metrica
        """,
        "bar",
    ),
    (
        "6. Segmentos de Cobranza por Período",
        """
        SELECT periodo, segmento_cobranza, n_obligaciones, pct_del_total
        FROM v_debit_risk_segment_summary
        ORDER BY periodo, segmento_cobranza
        """,
        "bar",
    ),
    (
        "7. KPI Resumen — Último Período",
        """
        SELECT periodo, total_obligaciones, pct_clase1, pct_clase0
        FROM v_debit_kpi_period
        ORDER BY periodo DESC
        LIMIT 1
        """,
        "scalar",
    ),
    (
        "8. Comparativa AUC y KS por Modelo",
        """
        WITH latest AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY model_name, split_strategy, feature_set
                       ORDER BY run_date DESC
                   ) AS rn
            FROM debit_model_metrics
            WHERE split_strategy != 'unknown'
        )
        SELECT model_name,
               split_strategy,
               feature_set,
               ROUND(cv_auc::numeric,    4) AS cv_auc,
               ROUND(train_auc::numeric, 4) AS train_auc,
               ROUND(test_auc::numeric,  4) AS test_auc,
               ROUND(oot_auc::numeric,   4) AS oot_auc,
               ROUND(test_ks::numeric,   4) AS test_ks,
               ROUND(oot_ks::numeric,    4) AS oot_ks
        FROM latest
        WHERE rn = 1
        ORDER BY feature_set, split_strategy, test_auc DESC
        """,
        "table",
    ),
    (
        "9. Estabilidad Train vs Test vs OOT por Modelo",
        """
        WITH latest AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY model_name, split_strategy, feature_set
                       ORDER BY run_date DESC
                   ) AS rn
            FROM debit_model_metrics
            WHERE split_strategy != 'unknown'
        ),
        base AS (
            SELECT model_name || ' [' || split_strategy || ' | opt' || feature_set || ']' AS modelo_config,
                   model_name, split_strategy, feature_set,
                   train_auc, test_auc, oot_auc
            FROM latest WHERE rn = 1
        )
        SELECT modelo_config, model_name, split_strategy, feature_set,
               'train' AS particion, ROUND(train_auc::numeric, 4) AS auc
        FROM base
        UNION ALL
        SELECT modelo_config, model_name, split_strategy, feature_set,
               'test',  ROUND(test_auc::numeric,  4)
        FROM base
        UNION ALL
        SELECT modelo_config, model_name, split_strategy, feature_set,
               'oot',   ROUND(oot_auc::numeric,   4)
        FROM base
        ORDER BY modelo_config, particion
        """,
        "bar",
        # X = modelo_config, series = particion, Y = auc
        {"graph.dimensions": ["modelo_config", "particion"], "graph.metrics": ["auc"]},
    ),
    (
        "10. Top 20 Features por Modelo",
        """
        WITH latest_runs AS (
            SELECT run_id, model_name, split_strategy, feature_set,
                   ROW_NUMBER() OVER (
                       PARTITION BY model_name, split_strategy, feature_set
                       ORDER BY run_date DESC
                   ) AS rn
            FROM debit_model_metrics
            WHERE split_strategy != 'unknown'
        ),
        latest AS (SELECT run_id, model_name, split_strategy, feature_set FROM latest_runs WHERE rn = 1)
        SELECT f.model_name || ' [opt' || l.feature_set || ']' AS modelo_config,
               f.model_name,
               l.split_strategy,
               l.feature_set,
               f.feature_name,
               ROUND(
                   (
                       (f.importance - MIN(f.importance) OVER (PARTITION BY f.model_name, l.feature_set)) /
                       NULLIF(
                           MAX(f.importance) OVER (PARTITION BY f.model_name, l.feature_set)
                           - MIN(f.importance) OVER (PARTITION BY f.model_name, l.feature_set),
                           0
                       )
                   )::numeric
               , 4) AS importance_norm
        FROM debit_model_features f
        JOIN latest l ON f.run_id = l.run_id
        WHERE f.rank <= 20
        ORDER BY f.model_name, l.feature_set, f.rank
        """,
        "bar",
        # X = feature_name, series = modelo_config, Y = importance_norm
        {"graph.dimensions": ["feature_name", "modelo_config"], "graph.metrics": ["importance_norm"]},
    ),
    # ── Preprocesamiento ────────────────────────────────────────────────────────
    (
        "11. Retención de Datos en Carga por Fuente",
        """
        SELECT fuente,
               rows_bruto::int    AS filas_brutas,
               rows_cargados::int AS filas_cargadas,
               retencion_pct
        FROM v_debit_pipeline_load_summary
        WHERE run_id = (
            SELECT run_id FROM debit_pipeline_metrics
            ORDER BY run_date DESC LIMIT 1
        )
        ORDER BY fuente
        """,
        "bar",
    ),
    (
        "12. Duplicados Eliminados en Carga",
        """
        SELECT fuente,
               COALESCE(dup_exactos::int,    0) AS duplicados_exactos,
               COALESCE(dup_conflicto::int,  0) AS duplicados_conflicto
        FROM v_debit_pipeline_load_summary
        WHERE run_id = (
            SELECT run_id FROM debit_pipeline_metrics
            ORDER BY run_date DESC LIMIT 1
        )
        ORDER BY fuente
        """,
        "bar",
    ),
    (
        "13. Funnel de Selección de Features",
        """
        WITH latest_runs AS (
            SELECT split_strategy, run_id,
                   ROW_NUMBER() OVER (PARTITION BY split_strategy ORDER BY MAX(run_date) DESC) AS rn
            FROM debit_pipeline_metrics
            GROUP BY split_strategy, run_id
        ),
        lr AS (SELECT split_strategy, run_id FROM latest_runs WHERE rn = 1)
        SELECT t.split_strategy, t.paso, t.cols::int AS n_features
        FROM (
            SELECT f.split_strategy, '1. Candidatas iniciales'   AS paso, cols_inicial::int      AS cols FROM v_debit_pipeline_feature_funnel f JOIN lr ON f.run_id = lr.run_id
            UNION ALL
            SELECT f.split_strategy, '2. Tras varianza/sparsity',         cols_tras_varianza::int        FROM v_debit_pipeline_feature_funnel f JOIN lr ON f.run_id = lr.run_id
            UNION ALL
            SELECT f.split_strategy, '3. Tras ANOVA F-test',              cols_tras_anova::int           FROM v_debit_pipeline_feature_funnel f JOIN lr ON f.run_id = lr.run_id
            UNION ALL
            SELECT f.split_strategy, '4. Final — ElasticNet',             cols_final::int                FROM v_debit_pipeline_feature_funnel f JOIN lr ON f.run_id = lr.run_id
        ) t
        ORDER BY t.split_strategy, t.paso
        """,
        "bar",
        {"graph.dimensions": ["split_strategy", "paso"], "graph.metrics": ["n_features"]},
    ),
    (
        "14. Dimensiones de Particiones Train / Test / OOT",
        """
        WITH latest_runs AS (
            SELECT split_strategy, run_id,
                   ROW_NUMBER() OVER (PARTITION BY split_strategy ORDER BY MAX(run_date) DESC) AS rn
            FROM debit_pipeline_metrics
            GROUP BY split_strategy, run_id
        ),
        lr AS (SELECT split_strategy, run_id FROM latest_runs WHERE rn = 1)
        SELECT s.split_strategy,
               s.particion,
               s.filas,
               s.features_candidatas,
               s.features_seleccionadas,
               s.tasa_clase1,
               s.memoria_mb,
               s.nulos_totales,
               s.inf_totales
        FROM v_debit_pipeline_splits s
        JOIN lr ON s.run_id = lr.run_id
        ORDER BY s.split_strategy, s.particion
        """,
        "table",
    ),
    (
        "15. Balance de Clases por Partición",
        """
        WITH latest_runs AS (
            SELECT split_strategy, run_id,
                   ROW_NUMBER() OVER (PARTITION BY split_strategy ORDER BY MAX(run_date) DESC) AS rn
            FROM debit_pipeline_metrics
            GROUP BY split_strategy, run_id
        ),
        lr AS (SELECT split_strategy, run_id FROM latest_runs WHERE rn = 1)
        SELECT s.split_strategy,
               s.particion,
               s.clase1_count AS clase_1_debito_recurrente,
               s.clase0_count AS clase_0_sin_patron
        FROM v_debit_pipeline_splits s
        JOIN lr ON s.run_id = lr.run_id
        ORDER BY s.split_strategy, s.particion
        """,
        "bar",
        {"graph.dimensions": ["split_strategy", "particion"], "graph.metrics": ["clase_1_debito_recurrente", "clase_0_sin_patron"]},
    ),
    (
        "16. Features Seleccionadas por Grupo Temático",
        """
        WITH latest_runs AS (
            SELECT split_strategy, run_id,
                   ROW_NUMBER() OVER (PARTITION BY split_strategy ORDER BY MAX(run_date) DESC) AS rn
            FROM debit_pipeline_metrics
            GROUP BY split_strategy, run_id
        ),
        lr AS (SELECT split_strategy, run_id FROM latest_runs WHERE rn = 1)
        SELECT pm.split_strategy,
               pm.source_name   AS grupo_tematico,
               pm.metric_value::int AS n_features
        FROM debit_pipeline_metrics pm
        JOIN lr ON pm.run_id = lr.run_id
        WHERE pm.stage       = 'feature_selection'
          AND pm.metric_name = 'cols_group'
        ORDER BY pm.split_strategy, pm.metric_value DESC
        """,
        "bar",
        {"graph.dimensions": ["split_strategy", "grupo_tematico"], "graph.metrics": ["n_features"]},
    ),
    (
        "17. Última Fecha de Carga de Datos",
        """
        SELECT MAX(run_date) AT TIME ZONE 'America/Bogota' AS ultima_carga_datos
        FROM debit_pipeline_metrics
        WHERE stage = 'loader'
        """,
        "scalar",
    ),
    (
        "18. Última Fecha de Entrenamiento",
        """
        SELECT MAX(run_date) AT TIME ZONE 'America/Bogota' AS ultimo_entrenamiento
        FROM debit_model_metrics
        """,
        "scalar",
    ),
]


# ─── Helpers HTTP ─────────────────────────────────────────────────────────────


def _get(url: str, token: str) -> dict | list:
    resp = requests.get(url, headers={"X-Metabase-Session": token}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _post(url: str, payload: dict, token: str = "") -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Metabase-Session"] = token
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _put(url: str, payload: dict, token: str) -> dict:
    headers = {"Content-Type": "application/json", "X-Metabase-Session": token}
    resp = requests.put(url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ─── Lógica de provisionamiento ───────────────────────────────────────────────


def wait_for_metabase() -> None:
    """Esperar hasta que Metabase responda /api/health con status ok."""
    for attempt in range(1, HEALTH_RETRIES + 1):
        try:
            r = requests.get(f"{METABASE_URL}/api/health", timeout=10)
            if r.status_code == 200 and r.json().get("status") == "ok":
                logger.info("Metabase disponible (intento %d).", attempt)
                return
        except requests.RequestException:
            pass
        logger.info("Esperando Metabase... %d/%d", attempt, HEALTH_RETRIES)
        time.sleep(HEALTH_WAIT)
    raise RuntimeError("Metabase no disponible tras múltiples intentos.")


def get_session_token() -> str:
    """Obtener token de sesión de administrador."""
    data  = _post(f"{METABASE_URL}/api/session", {"username": MB_USER, "password": MB_PASS})
    token = data.get("id") or data.get("token", "")
    logger.info("Sesión Metabase obtenida.")
    return str(token)


def setup_new_instance(token: str) -> None:
    """Completar wizard de configuración si es una instalación nueva."""
    try:
        props = _get(f"{METABASE_URL}/api/session/properties", token)
        setup_token = props.get("setup-token") if isinstance(props, dict) else None
        if not setup_token:
            return
        logger.info("Instancia nueva — ejecutando setup inicial...")
        _post(
            f"{METABASE_URL}/api/setup",
            {
                "token":  setup_token,
                "user":   {
                    "email":      MB_USER,
                    "password":   MB_PASS,
                    "first_name": "Admin",
                    "last_name":  "Bancolombia",
                    "site_name":  "Bancolombia Débitos",
                },
                "prefs":    {"site_name": "Bancolombia Débitos", "allow_tracking": False},
                "database": None,
            },
        )
        time.sleep(5)
        logger.info("Setup completado.")
    except Exception as exc:
        logger.debug("Setup no necesario o ya completado: %s", exc)


def get_or_create_database(token: str) -> int:
    """Registrar debitdb en Metabase si no existe.

    Returns
    -------
    int
        ID interno de la base de datos en Metabase.
    """
    raw  = _get(f"{METABASE_URL}/api/database", token)
    dbs  = raw.get("data", []) if isinstance(raw, dict) else raw
    for db in dbs:
        if db.get("name") == DB_DISPLAY:
            db_id = int(db["id"])
            logger.info("Conexión '%s' ya existe (id=%d).", DB_DISPLAY, db_id)
            return db_id

    result = _post(
        f"{METABASE_URL}/api/database",
        {
            "engine":  "postgres",
            "name":    DB_DISPLAY,
            "details": {
                "host":     DB_HOST,
                "port":     DB_PORT,
                "dbname":   DB_NAME,
                "user":     DB_USER,
                "password": DB_PASS,
            },
            "auto_run_queries":             True,
            "is_full_sync":                 True,
            "is_on_demand":                 False,
            "schedules":                    {},
            "metadata_sync_schedule":       "0 * * * *",
            "cache_field_values_schedule":  "0 0 * * *",
        },
        token,
    )
    db_id = int(result["id"])
    logger.info("Conexión '%s' creada (id=%d).", DB_DISPLAY, db_id)
    return db_id


def _find_card(title: str, token: str) -> int | None:
    raw   = _get(f"{METABASE_URL}/api/card?f=all", token)
    cards = raw if isinstance(raw, list) else raw.get("data", [])
    for c in cards:
        if c.get("name") == title:
            return int(c["id"])
    return None


def create_question(
    title: str, sql: str, viz_type: str, db_id: int, token: str,
    viz_settings: dict | None = None,
) -> int:
    """Crear o actualizar una question nativa SQL.

    Parameters
    ----------
    viz_settings : dict | None
        visualization_settings de Metabase. Usar graph.dimensions / graph.metrics
        para evitar el prompt de ejes en bar/line charts con múltiples columnas.

    Returns
    -------
    int
        ID de la card.
    """
    payload = {
        "name":          title,
        "display":       viz_type,
        "dataset_query": {
            "database": db_id,
            "type":     "native",
            "native":   {"query": sql.strip()},
        },
        "visualization_settings": viz_settings or {},
    }

    existing = _find_card(title, token)
    if existing:
        _put(f"{METABASE_URL}/api/card/{existing}", payload, token)
        logger.info("Card '%s' actualizada (id=%d).", title, existing)
        return existing

    result  = _post(f"{METABASE_URL}/api/card", payload, token)
    cid = int(result["id"])
    logger.info("Card '%s' creada (id=%d).", title, cid)
    return cid


def _find_dashboard(name: str, token: str) -> int | None:
    raw = _get(f"{METABASE_URL}/api/dashboard", token)
    lst = raw if isinstance(raw, list) else []
    for d in lst:
        if d.get("name") == name:
            return int(d["id"])
    return None


def create_dashboard(card_ids: list[int], token: str) -> int:
    """Crear (o actualizar) dashboard con las cards en layout 2 columnas.

    Siempre sincroniza el conjunto completo de cards aunque el dashboard ya exista.

    Returns
    -------
    int
        ID del dashboard.
    """
    existing = _find_dashboard(DASHBOARD_NAME, token)
    if existing:
        dash_id = existing
        logger.info("Dashboard '%s' ya existe (id=%d) — actualizando cards.", DASHBOARD_NAME, dash_id)
    else:
        result = _post(
            f"{METABASE_URL}/api/dashboard",
            {
                "name":        DASHBOARD_NAME,
                "description": "Tablero analítico de débitos recurrentes — mora temprana 1-30 días.",
            },
            token,
        )
        dash_id = int(result["id"])
        logger.info("Dashboard creado (id=%d).", dash_id)

    cards_payload = []
    for idx, cid in enumerate(card_ids):
        col = (idx % 2) * 12
        row = (idx // 2) * 8
        cards_payload.append({
            "id":                      -(idx + 1),
            "card_id":                 cid,
            "col":                     col,
            "row":                     row,
            "size_x":                  12,
            "size_y":                  8,
            "parameter_mappings":      [],
            "visualization_settings":  {},
        })

    resp = requests.put(
        f"{METABASE_URL}/api/dashboard/{dash_id}/cards",
        json={"cards": cards_payload},
        headers={"Content-Type": "application/json", "X-Metabase-Session": token},
        timeout=30,
    )
    resp.raise_for_status()
    logger.info("%d cards sincronizadas en dashboard id=%d.", len(card_ids), dash_id)

    return dash_id


# ─── Entrypoint ───────────────────────────────────────────────────────────────


def run_provisioning() -> None:
    """Pipeline completo de provisionamiento del tablero Metabase."""
    wait_for_metabase()

    try:
        token = get_session_token()
    except Exception:
        logger.info("Primera ejecución — configurando instancia nueva...")
        setup_new_instance("")
        # Esperar a que Metabase procese el setup y habilite el login
        for i in range(1, 13):
            time.sleep(10)
            try:
                token = get_session_token()
                break
            except Exception:
                logger.info("Esperando activación post-setup... %d/12", i)
        else:
            raise RuntimeError("No se pudo autenticar tras el setup de Metabase.")

    db_id    = get_or_create_database(token)
    card_ids = [
        create_question(t, sql, viz, db_id, token, entry[3] if len(entry) > 3 else None)
        for entry in _QUESTIONS
        for t, sql, viz in [(entry[0], entry[1], entry[2])]
    ]
    dash_id  = create_dashboard(card_ids, token)

    logger.info(
        "Provisionamiento completo | dashboard_id=%d | %s/dashboard/%d",
        dash_id, METABASE_URL, dash_id,
    )


if __name__ == "__main__":
    run_provisioning()
