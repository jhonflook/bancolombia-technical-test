"""Visualizaciones EDA para el análisis de débitos recurrentes — Bancolombia.

Produce gráficos orientados a decisiones de negocio de cobranza:
  - Distribución del target y evolución temporal
  - Mix de canales de pago por clase
  - Perfil de mora por segmento
  - Efectividad de gestiones de cobranza
  - Curvas ROC / Precision-Recall / KS del modelo
  - Feature importance

Todas las funciones retornan un plt.Figure y aceptan save_path opcional.

Supuestos:
  S8  Desbalance 78.8% clase 1 / 21.2% clase 0 — reflejado en paletas y anotaciones.
  S9  pago_debito_* usado solo en EDA descriptivo, no para evaluar leakage aquí.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc

plt.style.use("seaborn-v0_8-whitegrid")

_PALETTE = {"clase1": "#1f77b4", "clase0": "#d62728"}
_SEGMENT_COLORS = {
    "A - Automatizar":       "#2ca02c",
    "B - Monitorear":        "#ff7f0e",
    "C - Cobranza suave":    "#1f77b4",
    "D - Cobranza intensiva":"#d62728",
}


def _save(fig: plt.Figure, save_path: Optional[str]) -> None:
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


# ─── 1. Distribución del target ──────────────────────────────────────────────


def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str = "var_rta",
    figsize: tuple = (8, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Distribución absoluta y relativa de var_rta (clase 0 vs clase 1).

    Parameters
    ----------
    df : pd.DataFrame
        Modelo analítico con columna var_rta.
    target_col : str
        Nombre de la columna target.
    figsize : tuple
        Tamaño de la figura.
    save_path : str, optional
        Ruta para guardar la imagen.

    Returns
    -------
    plt.Figure
    """
    counts = df[target_col].value_counts().sort_index()
    pcts   = counts / counts.sum() * 100

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Absoluto
    bars = axes[0].bar(
        ["Clase 0\n(Otros canales)", "Clase 1\n(Débito exclusivo)"],
        counts.values,
        color=[_PALETTE["clase0"], _PALETTE["clase1"]],
        edgecolor="white",
    )
    for bar, cnt in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                     f"{cnt:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[0].set_title("Distribución absoluta", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Obligaciones")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # Porcentual — pie
    axes[1].pie(
        pcts.values,
        labels=[f"Clase 0\n{pcts.iloc[0]:.1f}%", f"Clase 1\n{pcts.iloc[1]:.1f}%"],
        colors=[_PALETTE["clase0"], _PALETTE["clase1"]],
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 11},
    )
    axes[1].set_title("Distribución relativa", fontsize=12, fontweight="bold")

    fig.suptitle("Variable respuesta: débito exclusivo recurrente (≥40%)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ─── 2. Evolución temporal del target ────────────────────────────────────────


def plot_temporal_evolution(
    df: pd.DataFrame,
    target_col: str = "var_rta",
    period_col: str = "f_analisis",
    figsize: tuple = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Porcentaje de clase 1 y volumen total por período de análisis.

    Parameters
    ----------
    df : pd.DataFrame
        Modelo analítico.
    target_col : str
        Columna target.
    period_col : str
        Columna datetime de período.
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    plt.Figure
    """
    agg = (
        df.groupby(period_col)[target_col]
        .agg(total="count", clase1="sum")
        .reset_index()
    )
    agg["pct_clase1"] = 100.0 * agg["clase1"] / agg["total"]
    agg[period_col] = pd.to_datetime(agg[period_col])

    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.bar(agg[period_col], agg["total"], width=20, color="#aec7e8",
            alpha=0.7, label="Total obligaciones")
    ax1.set_ylabel("Total obligaciones", fontsize=11)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    ax2 = ax1.twinx()
    ax2.plot(agg[period_col], agg["pct_clase1"], "o-", color=_PALETTE["clase1"],
             linewidth=2.5, markersize=6, label="% Clase 1")
    ax2.set_ylabel("% Débito exclusivo (Clase 1)", fontsize=11, color=_PALETTE["clase1"])
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis="y", labelcolor=_PALETTE["clase1"])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    ax1.set_title("Evolución temporal: volumen y tasa de débito exclusivo",
                  fontsize=13, fontweight="bold")
    ax1.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ─── 3. Mix de canales de pago ────────────────────────────────────────────────


def plot_payment_channels(
    df: pd.DataFrame,
    target_col: str = "var_rta",
    window: str = "6m",
    figsize: tuple = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Comparación del mix de canales de pago entre clase 0 y clase 1.

    Parameters
    ----------
    df : pd.DataFrame
        Modelo analítico con columnas avg_pago_*_<window>.
    target_col : str
    window : str
        Ventana temporal ('3m', '6m', '9m', '12m').
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    plt.Figure
    """
    canales = ["debito", "fisico", "virtual", "otros"]
    cols    = [f"avg_pago_{c}_{window}" for c in canales]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, f"Columnas no disponibles: {missing}", ha="center", va="center")
        return fig

    agg = df.groupby(target_col)[cols].mean().reset_index()
    agg.columns = [target_col] + canales

    x   = np.arange(len(canales))
    w   = 0.35
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Barras agrupadas
    ax = axes[0]
    for i, (_, row) in enumerate(agg.iterrows()):
        label = f"Clase {int(row[target_col])}"
        color = _PALETTE["clase1"] if int(row[target_col]) == 1 else _PALETTE["clase0"]
        ax.bar(x + i * w, row[canales].values, w, label=label, color=color, alpha=0.85)
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels([c.capitalize() for c in canales])
    ax.set_title(f"Pagos promedio por canal ({window})", fontsize=12, fontweight="bold")
    ax.set_ylabel("Monto promedio")
    ax.legend()

    # Stacked proporción
    ax2 = axes[1]
    totals = agg[canales].sum(axis=1).replace(0, np.nan)
    props  = agg[canales].div(totals, axis=0) * 100
    props.index = [f"Clase {int(r)}" for r in agg[target_col]]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]
    props.T.plot(kind="bar", stacked=False, ax=ax2, color=colors, alpha=0.85)
    ax2.set_title(f"Proporción de canales ({window})", fontsize=12, fontweight="bold")
    ax2.set_ylabel("% del total de pagos")
    ax2.set_xticklabels([c.capitalize() for c in canales], rotation=0)
    ax2.legend(title="Clase")

    fig.suptitle("Análisis de canales de pago por clase de débito recurrente",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ─── 4. Mora por clase ────────────────────────────────────────────────────────


def plot_mora_by_class(
    df: pd.DataFrame,
    target_col: str = "var_rta",
    mora_col: str = "moras_avg_mora_6m",
    figsize: tuple = (10, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Distribución de días de mora por clase (boxplot + violín).

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    mora_col : str
        Columna de días de mora a graficar.
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    plt.Figure
    """
    if mora_col not in df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Columna no disponible: {mora_col}", ha="center", va="center")
        return fig

    plot_df = df[[target_col, mora_col]].dropna()
    plot_df["Clase"] = plot_df[target_col].map({0: "Clase 0 (Otros)", 1: "Clase 1 (Débito)"})

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Violín
    sns.violinplot(
        data=plot_df, x="Clase", y=mora_col,
        palette=[_PALETTE["clase0"], _PALETTE["clase1"]],
        inner="quartile", ax=axes[0],
    )
    axes[0].set_title("Distribución de mora (violín)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Días de mora promedio 6m")
    axes[0].set_xlabel("")

    # Estadísticas descriptivas
    stats = plot_df.groupby("Clase")[mora_col].describe()[["mean", "50%", "75%", "max"]]
    stats.columns = ["Media", "Mediana", "P75", "Máx"]
    axes[1].axis("off")
    tbl = axes[1].table(
        cellText=[[f"{v:.2f}" for v in row] for row in stats.values],
        rowLabels=stats.index.tolist(),
        colLabels=stats.columns.tolist(),
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.8)
    axes[1].set_title("Estadísticos de mora por clase", fontsize=12, fontweight="bold")

    fig.suptitle("Perfil de mora: clase 0 vs clase 1", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ─── 5. Gestiones de cobranza por clase ──────────────────────────────────────


def plot_gestiones_profile(
    df: pd.DataFrame,
    target_col: str = "var_rta",
    window: str = "6m",
    figsize: tuple = (14, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Comparación de indicadores de cobranza entre clase 0 y clase 1.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    window : str
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    plt.Figure
    """
    indicadores = {
        f"avg_cant_gestiones_{window}":    "Gestiones",
        f"avg_cant_rpc_{window}":          "RPC",
        f"avg_cant_acuerdo_{window}":      "Acuerdos",
        f"avg_promesas_cumplidas_{window}": "Promesas\ncumplidas",
    }
    cols_present = {k: v for k, v in indicadores.items() if k in df.columns}
    if not cols_present:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Columnas de gestiones no disponibles", ha="center", va="center")
        return fig

    agg = df.groupby(target_col)[list(cols_present.keys())].mean()

    x  = np.arange(len(cols_present))
    w  = 0.35
    fig, ax = plt.subplots(figsize=figsize)

    for i, clase in enumerate(agg.index):
        color = _PALETTE["clase1"] if int(clase) == 1 else _PALETTE["clase0"]
        label = f"Clase {int(clase)}"
        ax.bar(x + i * w, agg.loc[clase].values, w, label=label, color=color, alpha=0.85)

    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(list(cols_present.values()))
    ax.set_title(f"Indicadores de gestión de cobranza ({window}) por clase",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Promedio")
    ax.legend()
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ─── 6. Segmentos de cobranza ─────────────────────────────────────────────────


def plot_risk_segments(
    df: pd.DataFrame,
    segment_col: str = "segmento_cobranza",
    figsize: tuple = (10, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Distribución de obligaciones por segmento de cobranza.

    Parameters
    ----------
    df : pd.DataFrame
        Debe contener columna segmento_cobranza (generada en EDA o en vista SQL).
    segment_col : str
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    plt.Figure
    """
    if segment_col not in df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Columna 'segmento_cobranza' no encontrada", ha="center", va="center")
        return fig

    counts = df[segment_col].value_counts().sort_index()
    colors = [_SEGMENT_COLORS.get(s, "#7f7f7f") for s in counts.index]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    bars = axes[0].barh(counts.index, counts.values, color=colors, edgecolor="white")
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_width() + 50, bar.get_y() + bar.get_height() / 2,
                     f"{val:,}", va="center", fontsize=9)
    axes[0].set_title("Obligaciones por segmento", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Obligaciones")
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    pcts = counts / counts.sum() * 100
    axes[1].pie(
        pcts.values, labels=[f"{s}\n{p:.1f}%" for s, p in zip(pcts.index, pcts.values)],
        colors=colors, startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 9},
    )
    axes[1].set_title("Distribución relativa", fontsize=12, fontweight="bold")

    fig.suptitle("Segmentación de cobranza: A=Automatizar | B=Monitorear | C=Suave | D=Intensiva",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ─── 7. Feature importance ────────────────────────────────────────────────────


def plot_feature_importance(
    importance_df: pd.DataFrame,
    title: str = "Feature Importance — Top 30",
    top_n: int = 30,
    figsize: tuple = (12, 9),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Gráfico horizontal de importancia de features del modelo XGBoost.

    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame con columnas 'feature' e 'importance'.
    title : str
    top_n : int
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    plt.Figure
    """
    if importance_df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No hay datos de importancia", ha="center", va="center")
        return fig

    top = importance_df.nlargest(top_n, "importance")
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(top)))

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(top["feature"], top["importance"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Importancia (gain)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ─── 8. Curvas de rendimiento del modelo ─────────────────────────────────────


def plot_model_performance(
    metrics_dict: dict[str, dict],
    figsize: tuple = (15, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Curvas ROC, Precision-Recall y distribución de scores por partición.

    Parameters
    ----------
    metrics_dict : dict
        Estructura: {'train': {'y_true': array, 'y_prob': array},
                     'test':  {...}, 'oot': {...}}
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    plt.Figure

    Notes
    -----
    S8: AUC-ROC es la métrica primaria; KS complementa la evaluación operativa.
    """
    colors     = {"train": "#aec7e8", "test": "#1f77b4", "oot": "#d62728"}
    linestyles = {"train": "--",      "test": "-",       "oot": "-."}

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for partition, data in metrics_dict.items():
        y_true = np.asarray(data["y_true"])
        y_prob = np.asarray(data["y_prob"])
        color  = colors.get(partition, "gray")
        ls     = linestyles.get(partition, "-")

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc     = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, color=color, ls=ls, lw=2,
                     label=f"{partition} (AUC={roc_auc:.3f})")

        # Precision-Recall
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        pr_auc       = auc(rec, prec)
        axes[1].plot(rec, prec, color=color, ls=ls, lw=2,
                     label=f"{partition} (AUC-PR={pr_auc:.3f})")

        # Score distribution
        axes[2].hist(y_prob[y_true == 0], bins=40, alpha=0.5, color=_PALETTE["clase0"],
                     density=True, label=f"{partition} Clase 0" if partition == "test" else None)
        axes[2].hist(y_prob[y_true == 1], bins=40, alpha=0.5, color=_PALETTE["clase1"],
                     density=True, label=f"{partition} Clase 1" if partition == "test" else None)

    # Decorar ROC
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(xlabel="FPR", ylabel="TPR", title="Curva ROC")
    axes[0].legend(fontsize=8)

    # Decorar PR
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Curva Precision-Recall")
    axes[1].legend(fontsize=8)

    # Decorar score dist
    axes[2].set(xlabel="Probabilidad predicha", ylabel="Densidad",
                title="Distribución de scores (Test)")
    axes[2].legend(fontsize=8)

    fig.suptitle("Rendimiento del modelo XGBoost — Train / Test / OOT",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ─── 9. Excedentes y % pago por clase ────────────────────────────────────────


def plot_excedentes_profile(
    df: pd.DataFrame,
    target_col: str = "var_rta",
    figsize: tuple = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Distribución de excedentes de pago y % pagado sobre cuota por clase.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    plt.Figure
    """
    cols_map = {
        "avg_porc_pago_3m":  "% pago 3m",
        "avg_porc_pago_6m":  "% pago 6m",
        "avg_porc_pago_12m": "% pago 12m",
    }
    present = {k: v for k, v in cols_map.items() if k in df.columns}
    if not present:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Columnas de excedentes no disponibles", ha="center", va="center")
        return fig

    agg  = df.groupby(target_col)[list(present.keys())].mean()
    x    = np.arange(len(present))
    w    = 0.35
    fig, ax = plt.subplots(figsize=figsize)

    for i, clase in enumerate(agg.index):
        color = _PALETTE["clase1"] if int(clase) == 1 else _PALETTE["clase0"]
        ax.bar(x + i * w, agg.loc[clase].values * 100, w,
               label=f"Clase {int(clase)}", color=color, alpha=0.85)

    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(list(present.values()))
    ax.set_title("Porcentaje promedio de pago sobre cuota por clase",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("% pagado sobre cuota")
    ax.legend()
    plt.tight_layout()
    _save(fig, save_path)
    return fig
