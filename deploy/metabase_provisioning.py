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
    METABASE_PASS=admin123 \
    DEBIT_DB_HOST=localhost \
    DEBIT_DB_NAME=debitdb \
    DEBIT_DB_USER=debit \
    DEBIT_DB_PASS=debit \
    uv run python deploy/metabase_provisioning.py

En Docker (llamado desde entrypoint.sh o DAG):
    python /app/deploy/metabase_provisioning.py
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
MB_PASS       = os.getenv("METABASE_PASS", "admin123")

DB_HOST       = os.getenv("DEBIT_DB_HOST", "postgres")
DB_NAME       = os.getenv("DEBIT_DB_NAME", "debitdb")
DB_USER       = os.getenv("DEBIT_DB_USER", "debit")
DB_PASS       = os.getenv("DEBIT_DB_PASS", "debit")
DB_PORT       = int(os.getenv("DEBIT_DB_PORT", "5432"))

DASHBOARD_NAME = "Débitos Recurrentes — Bancolombia"
DB_DISPLAY     = "debitdb (Débitos)"

HEALTH_RETRIES = 20
HEALTH_WAIT    = 15  # segundos


# ─── Definición de preguntas del tablero ──────────────────────────────────────

# Cada entrada: (nombre, query SQL, tipo_visualización)
_QUESTIONS: list[tuple[str, str, str]] = [
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
        "4. Mora Promedio 6m por Clase y Período",
        """
        SELECT periodo,
               MAX(CASE WHEN clase = 0 THEN avg_mora_6m END) AS mora_clase0,
               MAX(CASE WHEN clase = 1 THEN avg_mora_6m END) AS mora_clase1
        FROM v_debit_mora_profile
        GROUP BY periodo
        ORDER BY periodo
        """,
        "line",
    ),
    (
        "5. Mix de Canales de Pago por Clase",
        """
        SELECT clase,
               ROUND(AVG(avg_debito_6m),  4) AS debito_6m,
               ROUND(AVG(avg_fisico_6m),  4) AS fisico_6m,
               ROUND(AVG(avg_virtual_6m), 4) AS virtual_6m,
               ROUND(AVG(avg_otros_6m),   4) AS otros_6m
        FROM v_debit_payment_mix
        GROUP BY clase
        ORDER BY clase
        """,
        "bar",
    ),
    (
        "6. Efectividad de Gestiones de Cobranza por Clase",
        """
        SELECT clase,
               ROUND(AVG(avg_gestiones_6m),          3) AS gestiones,
               ROUND(AVG(avg_rpc_6m),                3) AS rpc,
               ROUND(AVG(avg_acuerdos_6m),           3) AS acuerdos,
               ROUND(AVG(avg_promesas_cumplidas_6m),  3) AS promesas_cumplidas
        FROM v_debit_gestiones_profile
        GROUP BY clase
        ORDER BY clase
        """,
        "bar",
    ),
    (
        "7. Porcentaje de Pago sobre Cuota por Período",
        """
        SELECT periodo,
               MAX(CASE WHEN clase = 0 THEN avg_porc_pago_6m END) AS pct_pago_clase0,
               MAX(CASE WHEN clase = 1 THEN avg_porc_pago_6m END) AS pct_pago_clase1
        FROM v_debit_excedentes_profile
        GROUP BY periodo
        ORDER BY periodo
        """,
        "line",
    ),
    (
        "8. Actividad Transaccional por Clase y Período",
        """
        SELECT periodo,
               MAX(CASE WHEN clase = 0 THEN pct_con_actividad END) AS actividad_clase0,
               MAX(CASE WHEN clase = 1 THEN pct_con_actividad END) AS actividad_clase1
        FROM v_debit_canal_activity
        GROUP BY periodo
        ORDER BY periodo
        """,
        "line",
    ),
    (
        "9. Segmentos de Cobranza por Período",
        """
        SELECT periodo, segmento_cobranza, n_obligaciones, pct_del_total
        FROM v_debit_risk_segment_summary
        ORDER BY periodo, segmento_cobranza
        """,
        "bar",
    ),
    (
        "10. KPI Resumen — Último Período",
        """
        SELECT periodo, total_obligaciones, pct_clase1, pct_clase0
        FROM v_debit_kpi_period
        ORDER BY periodo DESC
        LIMIT 1
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


def create_question(title: str, sql: str, viz_type: str, db_id: int, token: str) -> int:
    """Crear (o reusar) una question nativa SQL.

    Returns
    -------
    int
        ID de la card.
    """
    existing = _find_card(title, token)
    if existing:
        logger.info("Card '%s' ya existe (id=%d).", title, existing)
        return existing

    result  = _post(
        f"{METABASE_URL}/api/card",
        {
            "name":          title,
            "display":       viz_type,
            "dataset_query": {
                "database": db_id,
                "type":     "native",
                "native":   {"query": sql.strip()},
            },
            "visualization_settings": {},
        },
        token,
    )
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
    """Crear dashboard con las cards en layout 2 columnas (12 cols c/u, 8 filas c/u).

    Returns
    -------
    int
        ID del dashboard.
    """
    existing = _find_dashboard(DASHBOARD_NAME, token)
    if existing:
        logger.info("Dashboard '%s' ya existe (id=%d).", DASHBOARD_NAME, existing)
        return existing

    result  = _post(
        f"{METABASE_URL}/api/dashboard",
        {
            "name":        DASHBOARD_NAME,
            "description": "Tablero analítico de débitos recurrentes — mora temprana 1-30 días.",
        },
        token,
    )
    dash_id = int(result["id"])
    logger.info("Dashboard creado (id=%d).", dash_id)

    for idx, cid in enumerate(card_ids):
        col = (idx % 2) * 12
        row = (idx // 2) * 8
        _post(
            f"{METABASE_URL}/api/dashboard/{dash_id}/cards",
            {"cardId": cid, "col": col, "row": row, "size_x": 12, "size_y": 8},
            token,
        )
        logger.info("Card id=%d añadida (col=%d, row=%d).", cid, col, row)

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
        time.sleep(5)
        token = get_session_token()

    db_id    = get_or_create_database(token)
    card_ids = [create_question(t, sql, viz, db_id, token) for t, sql, viz in _QUESTIONS]
    dash_id  = create_dashboard(card_ids, token)

    logger.info(
        "Provisionamiento completo | dashboard_id=%d | %s/dashboard/%d",
        dash_id, METABASE_URL, dash_id,
    )


if __name__ == "__main__":
    run_provisioning()
