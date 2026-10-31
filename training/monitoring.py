from dataclasses import dataclass, field
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets.granulometria_dataset import TARGET_COLUMNS


@dataclass(frozen=True)
class RegressionMetrics:
    """Métricas de regresión expresadas sobre porcentajes de granulometría."""

    mse: float
    mae: float
    rmse: float
    r2: float
    mae_by_target: tuple[float, ...]


def calculate_regression_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> RegressionMetrics:
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if targets.shape != predictions.shape:
        raise ValueError("targets y predictions deben tener la misma forma.")
    if targets.ndim != 2 or targets.shape[1] != len(TARGET_COLUMNS):
        raise ValueError(
            f"Se esperaban matrices de forma (muestras, {len(TARGET_COLUMNS)})."
        )

    errors = predictions - targets
    squared_errors = errors**2
    target_variance = np.sum((targets - targets.mean(axis=0)) ** 2)
    residual_sum = np.sum(squared_errors)
    r2 = 1.0 - residual_sum / target_variance if target_variance else 0.0
    return RegressionMetrics(
        mse=float(np.mean(squared_errors)),
        mae=float(np.mean(np.abs(errors))),
        rmse=float(np.sqrt(np.mean(squared_errors))),
        r2=float(r2),
        mae_by_target=tuple(np.mean(np.abs(errors), axis=0).tolist()),
    )


@dataclass
class TrainingHistory:
    """Historial de métricas registrado durante un entrenamiento."""

    epochs: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)

    def add(
        self,
        epoch: int,
        train_loss: float,
        validation_loss: float,
        learning_rate: float,
    ) -> None:
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.validation_loss.append(validation_loss)
        self.learning_rates.append(learning_rate)


def save_history(history: TrainingHistory, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "validation_loss", "learning_rate"])
        writer.writerows(
            zip(
                history.epochs,
                history.train_loss,
                history.validation_loss,
                history.learning_rates,
            )
        )


def plot_loss_curves(
    history: TrainingHistory,
    output_path: str | Path,
    title: str = "Evolución de la pérdida",
) -> None:
    if not history.epochs:
        raise ValueError("No se puede graficar un historial vacío.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        history.epochs,
        history.train_loss,
        color="#1f4e79",
        linewidth=2,
        label="Entrenamiento",
    )
    axis.plot(
        history.epochs,
        history.validation_loss,
        color="#5b9bd5",
        linewidth=2,
        label="Validación",
    )
    axis.set_title(title, color="#17365d", fontweight="bold")
    axis.set_xlabel("Época")
    axis.set_ylabel("Pérdida MSE")
    axis.legend(frameon=True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_training_loss(
    history: TrainingHistory,
    output_path: str | Path,
    title: str = "Pérdida del entrenamiento final",
) -> None:
    if not history.epochs:
        raise ValueError("No se puede graficar un historial vacío.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        history.epochs,
        history.train_loss,
        color="#1f4e79",
        linewidth=2,
        label="Entrenamiento",
    )
    axis.set_title(title, color="#17365d", fontweight="bold")
    axis.set_xlabel("Época")
    axis.set_ylabel("Pérdida MSE")
    axis.legend(frameon=True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_metrics(
    metrics: RegressionMetrics,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerows(
            [
                ["mse", metrics.mse],
                ["mae", metrics.mae],
                ["rmse", metrics.rmse],
                ["r2", metrics.r2],
            ]
        )
        writer.writerow([])
        writer.writerow(["target", "mae"])
        writer.writerows(zip(TARGET_COLUMNS, metrics.mae_by_target))


def plot_prediction_diagnostics(
    targets: np.ndarray,
    predictions: np.ndarray,
    output_dir: str | Path,
    title_prefix: str = "Validación",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    flattened_targets = targets.ravel()
    flattened_predictions = predictions.ravel()
    low = min(flattened_targets.min(), flattened_predictions.min())
    high = max(flattened_targets.max(), flattened_predictions.max())

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(
        flattened_targets,
        flattened_predictions,
        s=18,
        alpha=0.55,
        color="#2f75b5",
        edgecolors="none",
    )
    axis.plot([low, high], [low, high], color="#17365d", linewidth=2)
    axis.set_title(f"{title_prefix}: valores reales y predichos", color="#17365d")
    axis.set_xlabel("Valor real (%)")
    axis.set_ylabel("Valor predicho (%)")
    figure.tight_layout()
    figure.savefig(output_dir / "reales_vs_predichos.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    metrics = calculate_regression_metrics(targets, predictions)
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(TARGET_COLUMNS, metrics.mae_by_target, color="#5b9bd5")
    axis.set_title(f"{title_prefix}: MAE por tamaño de partícula", color="#17365d")
    axis.set_xlabel("Tamaño de partícula (mm)")
    axis.set_ylabel("MAE (puntos porcentuales)")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(output_dir / "mae_por_tamano.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_curve_examples(
    targets: np.ndarray,
    predictions: np.ndarray,
    output_path: str | Path,
    examples: int = 5,
) -> None:
    if targets.shape != predictions.shape or targets.ndim != 2:
        raise ValueError("targets y predictions deben ser matrices compatibles.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 5))
    x_values = np.arange(len(TARGET_COLUMNS))
    count = min(examples, len(targets))
    for index in range(count):
        axis.plot(
            x_values,
            targets[index],
            color="#17365d",
            alpha=0.35,
            linewidth=1.5,
        )
        axis.plot(
            x_values,
            predictions[index],
            color="#5b9bd5",
            alpha=0.35,
            linewidth=1.5,
            linestyle="--",
        )
    axis.plot([], [], color="#17365d", label="Real")
    axis.plot([], [], color="#5b9bd5", linestyle="--", label="Predicho")
    axis.set_title("Ejemplos de curvas granulométricas", color="#17365d")
    axis.set_xlabel("Tamaño de partícula (mm)")
    axis.set_ylabel("Porcentaje (%)")
    axis.set_xticks(x_values, TARGET_COLUMNS, rotation=45)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
