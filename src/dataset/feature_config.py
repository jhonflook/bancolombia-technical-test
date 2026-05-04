"""Configuración de grupos de features para el modelo de débitos recurrentes."""

from dataclasses import dataclass, field

WINDOWS_FULL = ["3m", "6m", "9m", "12m"]
WINDOWS_MORA = ["3m", "6m", "9m"]  # S6: moras no tiene ventana 12m
STATS = ["avg", "min", "max", "stddev"]

# Opción B (H1): grupos cuya señal es determinista porque clase-0 = cero actividad
# de pago en todos los canales y ventanas (pagos, excedentes, canales).
# Solo sobreviven gestiones + moras, que capturan cobranza independientemente del pago.
OPTION_B_EXCLUDE_GROUPS = [
    "pagos",
    "derived_pagos",
    "excedentes",
    "derived_excedentes",
    "canales_summary",
    "derived_canales",
]


@dataclass
class DebitFeatureConfig:
    """Grupos de features por fuente del datalake.

    Cada atributo lista los nombres de columnas que el modelo analítico
    produce al hacer el join de las 6 fuentes CSV.

    Attributes
    ----------
    pagos : list[str]
        Columnas del archivo tanque (pagos por canal, 4 ventanas).
    gestiones : list[str]
        Columnas de gestiones de cobranza (4 ventanas).
    moras : list[str]
        Columnas de moras (ventanas 3m/6m/9m — S6).
    excedentes : list[str]
        Columnas de excedentes de pago (4 ventanas).
    canales_summary : list[str]
        Columnas de resumen del archivo de canales transaccionales.
    derived_pagos : list[str]
        Features derivadas de exclusividad y proporción de débito.
    derived_gestiones : list[str]
        Ratios derivados de gestiones de cobranza.
    derived_moras : list[str]
        Features de estabilidad de mora.
    derived_excedentes : list[str]
        Indicadores de sobrepago sobre cuota.
    derived_canales : list[str]
        Features de actividad transaccional (resumen canales).
    """

    pagos: list[str] = field(default_factory=lambda: [
        f"{stat}_pago_{canal}_{w}"
        for stat in STATS
        for canal in ["debito", "fisico", "virtual", "otros"]
        for w in WINDOWS_FULL
    ])

    gestiones: list[str] = field(default_factory=lambda: [
        f"{stat}_{var}_{w}"
        for stat in STATS
        for var in [
            "cant_gestiones", "cant_rpc", "cant_acuerdo",
            "promesas_cumplidas", "maximo_rank",
        ]
        for w in WINDOWS_FULL
    ])

    moras: list[str] = field(default_factory=lambda: [
        f"moras_{stat}_mora_{w}"
        for stat in STATS
        for w in WINDOWS_MORA
    ])

    excedentes: list[str] = field(default_factory=lambda: [
        f"{stat}_{var}_{w}"
        for stat in STATS
        for var in ["excedente_pago", "porc_pago"]
        for w in WINDOWS_FULL
    ])

    canales_summary: list[str] = field(default_factory=lambda: [
        "trx_cnt_total",
        "trx_mnt_total",
        "trx_mnt_total_smmlv",
    ])

    derived_pagos: list[str] = field(default_factory=lambda: [
        f"{feat}_{w}"
        for feat in [
            "prop_debito", "canal_unico", "has_debito",
            "debito_exclusivo_40pct", "cv_debito",
        ]
        for w in WINDOWS_FULL
    ])

    derived_gestiones: list[str] = field(default_factory=lambda: [
        f"{feat}_{w}"
        for feat in ["ratio_acuerdos", "tasa_rpc", "promesas_pct", "rank_norm"]
        for w in WINDOWS_FULL
    ])

    derived_moras: list[str] = field(default_factory=lambda: [
        f"{feat}_{w}"
        for feat in ["cv_mora", "rango_mora"]
        for w in WINDOWS_MORA
    ])

    derived_excedentes: list[str] = field(default_factory=lambda: [
        f"{feat}_{w}"
        for feat in ["has_excedente", "sobrepago"]
        for w in WINDOWS_FULL
    ])

    derived_canales: list[str] = field(default_factory=lambda: [
        "has_trx_canales",
        "trx_mnt_smmlv_log1p",
    ])

    def all_base_features(self) -> list[str]:
        """Features base del modelo analítico (sin derivadas)."""
        return (
            self.pagos
            + self.gestiones
            + self.moras
            + self.excedentes
            + self.canales_summary
        )

    def all_derived_features(self) -> list[str]:
        """Features derivadas construidas por feature_engineering."""
        return (
            self.derived_pagos
            + self.derived_gestiones
            + self.derived_moras
            + self.derived_excedentes
            + self.derived_canales
        )

    def all_features(self) -> list[str]:
        """Catálogo completo: base + derivadas."""
        return self.all_base_features() + self.all_derived_features()

    def features_option_b(self) -> list[str]:
        """Features Opción B: excluye grupos deterministas por H1 (clase-0 = sin pago).

        Retiene solo gestiones y moras, que capturan comportamiento de cobranza
        independientemente de si el cliente tiene actividad de pago registrada.
        """
        exclude = set(
            feat
            for g in OPTION_B_EXCLUDE_GROUPS
            for feat in self.get_group(g)
        )
        return [f for f in self.all_features() if f not in exclude]

    def get_group(self, group: str) -> list[str]:
        """Retornar features de un grupo específico.

        Parameters
        ----------
        group : str
            Uno de: 'pagos', 'gestiones', 'moras', 'excedentes',
            'canales_summary', 'derived_pagos', 'derived_gestiones',
            'derived_moras', 'derived_excedentes', 'derived_canales'.

        Returns
        -------
        list[str]
            Lista de nombres de features del grupo.

        Raises
        ------
        ValueError
            Si el nombre de grupo no existe.
        """
        if not hasattr(self, group):
            valid = (
                "pagos, gestiones, moras, excedentes, canales_summary, "
                "derived_pagos, derived_gestiones, derived_moras, "
                "derived_excedentes, derived_canales"
            )
            raise ValueError(f"Grupo '{group}' no existe. Disponibles: {valid}")
        return list(getattr(self, group))


DEFAULT_FEATURE_CONFIG = DebitFeatureConfig()
