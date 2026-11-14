"""Grafico del MAE global de las 4 configuraciones comparadas con validacion cruzada.

Uso (desde la raiz del proyecto):
    python scripts/grafico_comparacion_configuraciones.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "resultados" / "entrenamiento"
OUTPUT_PATH = RESULTS_DIR / "comparacion_configuraciones.png"

CONFIGURATIONS = [
    ("1. Backbone congelado\nlr 1e-3, 10 epocas", "monitoreo_validacion_cruzada"),
    ("2. Backbone congelado\nlr 1e-4, 20 epocas", "adamw_lr1e-4_20epocas"),
    ("3. Backbone congelado\nlr 1e-3, 20 epocas", "adamw_lr1e-3_20epocas"),
    ("4. Ultimo bloque descongelado\nlr 1e-3 / 1e-5, 20 epocas", "validacion_cruzada"),
]

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SELECTED = "#2a78d6"
OTHER = "#9ec5f4"
BASELINE = "#52514e"


def global_mae(folder: Path) -> float:
    metrics = pd.read_csv(folder / "metrics_summary.csv", nrows=4)
    return float(metrics.loc[metrics["metric"] == "mae", "value"].iloc[0])


def main() -> None:
    labels = [label for label, _ in CONFIGURATIONS]
    values = [global_mae(RESULTS_DIR / folder) for _, folder in CONFIGURATIONS]
    best = values.index(min(values))
    colors = [SELECTED if index == best else OTHER for index in range(len(values))]

    summary = pd.read_csv(RESULTS_DIR / "validacion_cruzada" / "resumen_por_fold.csv")
    baseline = float(
        summary.loc[summary["fold"].astype(str).str.startswith("global"), "baseline_mae"].iloc[0]
    )

    figure, axis = plt.subplots(figsize=(10, 5.4), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)

    bars = axis.bar(range(len(values)), values, width=0.55, color=colors, zorder=3)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.3,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=TEXT_PRIMARY,
        )

    axis.axhline(baseline, color=BASELINE, linewidth=1.5, linestyle="--", zorder=4)
    figure.text(
        0.015,
        0.925,
        f"Linea punteada: referencia sin mirar la imagen (predecir la curva promedio), MAE {baseline:.2f}",
        ha="left",
        va="top",
        fontsize=9,
        color=TEXT_SECONDARY,
    )

    axis.set_xticks(range(len(values)), labels, fontsize=9, color=TEXT_PRIMARY)
    axis.set_ylabel("MAE global (puntos porcentuales)", color=TEXT_SECONDARY)
    figure.text(
        0.015,
        0.975,
        "MAE global por configuracion en validacion cruzada de 5 folds (menor es mejor)",
        ha="left",
        va="top",
        fontsize=12,
        color=TEXT_PRIMARY,
    )
    axis.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    axis.tick_params(colors=TEXT_SECONDARY, length=0)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(GRID)
    axis.set_ylim(0, max(values) * 1.15)

    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(OUTPUT_PATH, dpi=160, facecolor=SURFACE)
    plt.close(figure)
    print(f"Grafico guardado en: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
