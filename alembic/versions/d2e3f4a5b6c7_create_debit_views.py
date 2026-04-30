"""create_debit_views

Revision ID: d2e3f4a5b6c7
Revises: c1d2e3f4a5b6
Create Date: 2026-04-30

Crea las 9 vistas analíticas para el tablero Metabase.
Depende de la migración c1d2e3f4a5b6 (tablas debit_*).
"""

from alembic import op

revision = "d2e3f4a5b6c7"
down_revision = "c1d2e3f4a5b6"
branch_labels = None
depends_on = None


_VIEWS = [
    "v_debit_kpi_period",
    "v_debit_payment_mix",
    "v_debit_mora_profile",
    "v_debit_gestiones_profile",
    "v_debit_risk_segments",
    "v_debit_risk_segment_summary",
    "v_debit_excedentes_profile",
    "v_debit_canal_activity",
    "v_debit_master",
]


def upgrade() -> None:
    op.execute("""
        CREATE OR REPLACE VIEW v_debit_kpi_period AS
        SELECT
            f_analisis::date                                                    AS periodo,
            COUNT(*)                                                            AS total_obligaciones,
            SUM(var_rta)                                                        AS obligaciones_clase1,
            COUNT(*) - SUM(var_rta)                                             AS obligaciones_clase0,
            ROUND(100.0 * SUM(var_rta) / NULLIF(COUNT(*), 0), 2)               AS pct_clase1,
            ROUND(100.0 * (COUNT(*) - SUM(var_rta)) / NULLIF(COUNT(*), 0), 2)  AS pct_clase0
        FROM debit_clients
        GROUP BY f_analisis
        ORDER BY f_analisis
    """)

    op.execute("""
        CREATE OR REPLACE VIEW v_debit_payment_mix AS
        SELECT
            c.f_analisis::date                                               AS periodo,
            c.var_rta                                                        AS clase,
            ROUND(AVG(COALESCE(p.avg_pago_debito_6m,  0))::numeric, 4)      AS avg_debito_6m,
            ROUND(AVG(COALESCE(p.avg_pago_fisico_6m,  0))::numeric, 4)      AS avg_fisico_6m,
            ROUND(AVG(COALESCE(p.avg_pago_virtual_6m, 0))::numeric, 4)      AS avg_virtual_6m,
            ROUND(AVG(COALESCE(p.avg_pago_otros_6m,   0))::numeric, 4)      AS avg_otros_6m,
            COUNT(*)                                                         AS n_obligaciones
        FROM debit_clients c
        LEFT JOIN debit_pagos p
               ON c.num_doc = p.num_doc AND c.obl17 = p.obl17 AND c.f_analisis = p.f_analisis
        GROUP BY c.f_analisis, c.var_rta
        ORDER BY c.f_analisis, c.var_rta
    """)

    op.execute("""
        CREATE OR REPLACE VIEW v_debit_mora_profile AS
        SELECT
            c.f_analisis::date                                          AS periodo,
            c.var_rta                                                   AS clase,
            ROUND(AVG(COALESCE(m.moras_avg_mora_3m, 0))::numeric, 2)   AS avg_mora_3m,
            ROUND(AVG(COALESCE(m.moras_avg_mora_6m, 0))::numeric, 2)   AS avg_mora_6m,
            ROUND(AVG(COALESCE(m.moras_max_mora_6m, 0))::numeric, 2)   AS max_mora_6m,
            ROUND(AVG(COALESCE(m.moras_avg_mora_9m, 0))::numeric, 2)   AS avg_mora_9m,
            COUNT(*)                                                    AS n_obligaciones
        FROM debit_clients c
        LEFT JOIN debit_moras m
               ON c.num_doc = m.num_doc AND c.obl17 = m.obl17 AND c.f_analisis = m.f_analisis
        GROUP BY c.f_analisis, c.var_rta
        ORDER BY c.f_analisis, c.var_rta
    """)

    op.execute("""
        CREATE OR REPLACE VIEW v_debit_gestiones_profile AS
        SELECT
            c.f_analisis::date                                                    AS periodo,
            c.var_rta                                                             AS clase,
            ROUND(AVG(COALESCE(g.avg_cant_gestiones_6m,     0))::numeric, 3)     AS avg_gestiones_6m,
            ROUND(AVG(COALESCE(g.avg_cant_rpc_6m,           0))::numeric, 3)     AS avg_rpc_6m,
            ROUND(AVG(COALESCE(g.avg_cant_acuerdo_6m,       0))::numeric, 3)     AS avg_acuerdos_6m,
            ROUND(AVG(COALESCE(g.avg_promesas_cumplidas_6m, 0))::numeric, 3)     AS avg_promesas_cumplidas_6m,
            ROUND(AVG(COALESCE(g.avg_maximo_rank_6m,        0))::numeric, 3)     AS avg_maximo_rank_6m,
            COUNT(*)                                                              AS n_obligaciones
        FROM debit_clients c
        LEFT JOIN debit_gestiones g
               ON c.num_doc = g.num_doc AND c.obl17 = g.obl17 AND c.f_analisis = g.f_analisis
        GROUP BY c.f_analisis, c.var_rta
        ORDER BY c.f_analisis, c.var_rta
    """)

    op.execute("""
        CREATE OR REPLACE VIEW v_debit_risk_segments AS
        SELECT
            c.num_doc,
            c.obl17,
            c.f_analisis::date                      AS periodo,
            c.var_rta,
            COALESCE(m.moras_avg_mora_6m,      0)   AS mora_avg_6m,
            COALESCE(g.avg_cant_gestiones_6m,  0)   AS gestiones_avg_6m,
            COALESCE(p.avg_pago_debito_6m,     0)   AS pago_debito_avg_6m,
            COALESCE(e.avg_porc_pago_6m,       0)   AS porc_pago_6m,
            CASE
                WHEN c.var_rta = 1 AND COALESCE(m.moras_avg_mora_6m, 0) <= 10
                    THEN 'A - Automatizar'
                WHEN c.var_rta = 1 AND COALESCE(m.moras_avg_mora_6m, 0) > 10
                    THEN 'B - Monitorear'
                WHEN c.var_rta = 0 AND COALESCE(m.moras_avg_mora_6m, 0) <= 15
                    THEN 'C - Cobranza suave'
                ELSE
                    'D - Cobranza intensiva'
            END AS segmento_cobranza
        FROM debit_clients c
        LEFT JOIN debit_moras      m  ON c.num_doc = m.num_doc  AND c.obl17 = m.obl17  AND c.f_analisis = m.f_analisis
        LEFT JOIN debit_gestiones  g  ON c.num_doc = g.num_doc  AND c.obl17 = g.obl17  AND c.f_analisis = g.f_analisis
        LEFT JOIN debit_pagos      p  ON c.num_doc = p.num_doc  AND c.obl17 = p.obl17  AND c.f_analisis = p.f_analisis
        LEFT JOIN debit_excedentes e  ON c.num_doc = e.num_doc  AND c.obl17 = e.obl17  AND c.f_analisis = e.f_analisis
    """)

    op.execute("""
        CREATE OR REPLACE VIEW v_debit_risk_segment_summary AS
        SELECT
            periodo,
            segmento_cobranza,
            COUNT(*)                                                                              AS n_obligaciones,
            ROUND(100.0 * COUNT(*) / NULLIF(SUM(COUNT(*)) OVER (PARTITION BY periodo), 0), 2)   AS pct_del_total,
            ROUND(AVG(mora_avg_6m)::numeric,      2)                                             AS mora_promedio_6m,
            ROUND(AVG(gestiones_avg_6m)::numeric, 2)                                             AS gestiones_promedio_6m,
            ROUND(AVG(porc_pago_6m)::numeric,     4)                                             AS pct_pago_promedio_6m
        FROM v_debit_risk_segments
        GROUP BY periodo, segmento_cobranza
        ORDER BY periodo, segmento_cobranza
    """)

    op.execute("""
        CREATE OR REPLACE VIEW v_debit_excedentes_profile AS
        SELECT
            c.f_analisis::date                                              AS periodo,
            c.var_rta                                                       AS clase,
            ROUND(AVG(COALESCE(e.avg_excedente_pago_3m, 0))::numeric, 2)   AS avg_excedente_3m,
            ROUND(AVG(COALESCE(e.avg_excedente_pago_6m, 0))::numeric, 2)   AS avg_excedente_6m,
            ROUND(AVG(COALESCE(e.avg_porc_pago_3m,      0))::numeric, 4)   AS avg_porc_pago_3m,
            ROUND(AVG(COALESCE(e.avg_porc_pago_6m,      0))::numeric, 4)   AS avg_porc_pago_6m,
            ROUND(AVG(COALESCE(e.avg_porc_pago_12m,     0))::numeric, 4)   AS avg_porc_pago_12m,
            COUNT(*)                                                        AS n_obligaciones
        FROM debit_clients c
        LEFT JOIN debit_excedentes e
               ON c.num_doc = e.num_doc AND c.obl17 = e.obl17 AND c.f_analisis = e.f_analisis
        GROUP BY c.f_analisis, c.var_rta
        ORDER BY c.f_analisis, c.var_rta
    """)

    op.execute("""
        CREATE OR REPLACE VIEW v_debit_canal_activity AS
        SELECT
            c.f_analisis::date                                                           AS periodo,
            c.var_rta                                                                    AS clase,
            COUNT(*)                                                                     AS total_obligaciones,
            SUM(CASE WHEN COALESCE(ca.trx_cnt_total, 0) > 0 THEN 1 ELSE 0 END)         AS con_actividad_canal,
            ROUND(
                100.0 * SUM(CASE WHEN COALESCE(ca.trx_cnt_total, 0) > 0 THEN 1 ELSE 0 END)
                / NULLIF(COUNT(*), 0), 2
            )                                                                            AS pct_con_actividad,
            ROUND(AVG(COALESCE(ca.trx_mnt_total,       0))::numeric, 2)                 AS avg_monto_total,
            ROUND(AVG(COALESCE(ca.trx_mnt_total_smmlv, 0))::numeric, 4)                 AS avg_monto_smmlv,
            ROUND(AVG(COALESCE(ca.trx_cnt_total,        0))::numeric, 2)                AS avg_cnt_transacciones
        FROM debit_clients c
        LEFT JOIN debit_canales ca
               ON c.num_doc = ca.num_doc AND c.obl17 = ca.obl17 AND c.f_analisis = ca.f_analisis
        GROUP BY c.f_analisis, c.var_rta
        ORDER BY c.f_analisis, c.var_rta
    """)

    op.execute("""
        CREATE OR REPLACE VIEW v_debit_master AS
        SELECT
            c.num_doc,
            c.obl17,
            c.f_analisis::date                              AS periodo,
            c.var_rta,
            COALESCE(m.moras_avg_mora_3m, 0)               AS mora_avg_3m,
            COALESCE(m.moras_avg_mora_6m, 0)               AS mora_avg_6m,
            COALESCE(m.moras_max_mora_6m, 0)               AS mora_max_6m,
            COALESCE(p.avg_pago_debito_6m,  0)             AS pago_debito_6m,
            COALESCE(p.avg_pago_fisico_6m,  0)             AS pago_fisico_6m,
            COALESCE(p.avg_pago_virtual_6m, 0)             AS pago_virtual_6m,
            COALESCE(p.avg_pago_otros_6m,   0)             AS pago_otros_6m,
            COALESCE(e.avg_porc_pago_6m,          0)       AS porc_pago_6m,
            COALESCE(e.avg_excedente_pago_6m,     0)       AS excedente_6m,
            COALESCE(g.avg_cant_gestiones_6m,     0)       AS gestiones_6m,
            COALESCE(g.avg_cant_acuerdo_6m,       0)       AS acuerdos_6m,
            COALESCE(g.avg_promesas_cumplidas_6m, 0)       AS promesas_cumplidas_6m,
            COALESCE(ca.trx_cnt_total,            0)       AS trx_cnt_total,
            COALESCE(ca.trx_mnt_total,            0)       AS trx_mnt_total,
            CASE
                WHEN c.var_rta = 1 AND COALESCE(m.moras_avg_mora_6m, 0) <= 10
                    THEN 'A - Automatizar'
                WHEN c.var_rta = 1 AND COALESCE(m.moras_avg_mora_6m, 0) > 10
                    THEN 'B - Monitorear'
                WHEN c.var_rta = 0 AND COALESCE(m.moras_avg_mora_6m, 0) <= 15
                    THEN 'C - Cobranza suave'
                ELSE
                    'D - Cobranza intensiva'
            END AS segmento_cobranza
        FROM debit_clients c
        LEFT JOIN debit_moras      m  ON c.num_doc = m.num_doc  AND c.obl17 = m.obl17  AND c.f_analisis = m.f_analisis
        LEFT JOIN debit_pagos      p  ON c.num_doc = p.num_doc  AND c.obl17 = p.obl17  AND c.f_analisis = p.f_analisis
        LEFT JOIN debit_excedentes e  ON c.num_doc = e.num_doc  AND c.obl17 = e.obl17  AND c.f_analisis = e.f_analisis
        LEFT JOIN debit_gestiones  g  ON c.num_doc = g.num_doc  AND c.obl17 = g.obl17  AND c.f_analisis = g.f_analisis
        LEFT JOIN debit_canales    ca ON c.num_doc = ca.num_doc AND c.obl17 = ca.obl17 AND c.f_analisis = ca.f_analisis
    """)


def downgrade() -> None:
    for view in reversed(_VIEWS):
        op.execute(f"DROP VIEW IF EXISTS {view} CASCADE")
