"""Validación cruzada agrupada de la configuración final.

Ejecuta la misma configuración usada en el entrenamiento final, pero evaluada
con K folds agrupados por sample_id. Los resultados se guardan en una carpeta
propia, separada del entrenamiento final, para tener respaldo de ambos.

Uso (desde la raíz del proyecto):
    python -m training.ejecutar_validacion_cruzada
    python -m training.ejecutar_validacion_cruzada --epochs 20 --n-splits 5
"""

from argparse import ArgumentParser, Namespace
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from datasets.granulometria_dataset import TARGET_COLUMNS
from training.cross_validation import create_grouped_folds
from training.train import CSV_PATH, IMAGES_DIR, OUTPUT_DIR, run_cross_validation


DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "validacion_cruzada"


def describe_folds(
    csv_path: Path,
    n_splits: int,
    random_state: int,
) -> tuple[pd.DataFrame, float]:
    """Cantidad de imágenes y muestras por fold, y MAE del baseline.

    El baseline predice siempre la curva promedio del training de cada fold,
    sin mirar la imagen. Devuelve la tabla por fold y el MAE global del
    baseline calculado sobre todas las imágenes de validación juntas.
    """
    dataframe = pd.read_csv(csv_path)
    folds = create_grouped_folds(dataframe, n_splits, random_state)
    rows = []
    all_errors = []
    for fold in folds:
        mean_curve = fold.train[TARGET_COLUMNS].to_numpy(dtype=float).mean(axis=0)
        targets = fold.validation[TARGET_COLUMNS].to_numpy(dtype=float)
        errors = np.abs(targets - mean_curve)
        all_errors.append(errors)
        rows.append(
            {
                "fold": fold.fold_number,
                "imagenes_train": len(fold.train),
                "muestras_train": fold.train["sample_id"].nunique(),
                "imagenes_validacion": len(fold.validation),
                "muestras_validacion": fold.validation["sample_id"].nunique(),
                "muestras_en_validacion": " ".join(sorted(fold.validation_ids)),
                "baseline_mae": float(errors.mean()),
            }
        )
    global_baseline = float(np.concatenate(all_errors).mean())
    return pd.DataFrame(rows), global_baseline


def read_global_metrics(path: Path) -> dict[str, float]:
    """Lee las metricas globales que guarda run_cross_validation."""
    metrics = {}
    with path.open(encoding="utf-8") as file:
        next(file)
        for line in file:
            line = line.strip()
            if not line:
                break
            name, value = line.split(",")
            metrics[name] = float(value)
    return metrics


def save_fold_summary(
    results: list,
    fold_info: pd.DataFrame,
    global_baseline: float,
    output_dir: Path,
) -> pd.DataFrame:
    summary = fold_info.copy()
    summary["mae"] = [metrics.mae for metrics in results]
    summary["rmse"] = [metrics.rmse for metrics in results]
    summary["mse"] = [metrics.mse for metrics in results]
    summary["r2"] = [metrics.r2 for metrics in results]

    metric_columns = ["mae", "rmse", "mse", "r2", "baseline_mae"]
    global_metrics = read_global_metrics(output_dir / "metrics_summary.csv")

    extra_rows = pd.DataFrame(
        [
            {
                "fold": "global (todas las imágenes juntas)",
                "imagenes_validacion": int(summary["imagenes_validacion"].sum()),
                "muestras_validacion": int(summary["muestras_validacion"].sum()),
                **{name: global_metrics[name] for name in ("mae", "rmse", "mse", "r2")},
                "baseline_mae": global_baseline,
            },
            {
                "fold": "promedio simple de los folds",
                **summary[metric_columns].mean().to_dict(),
            },
            {
                "fold": "desviación estándar entre folds",
                **summary[metric_columns].std().to_dict(),
            },
        ]
    )
    summary = pd.concat([summary, extra_rows], ignore_index=True)
    summary.to_csv(output_dir / "resumen_por_fold.csv", index=False)
    return summary


def parse_arguments() -> Namespace:
    parser = ArgumentParser(
        description="Validación cruzada agrupada de la configuración final."
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--optimizer", choices=("adamw", "adam"), default="adamw")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--unfreeze-last-blocks", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.epochs < 1:
        raise ValueError("--epochs debe ser mayor que cero.")

    output_dir: Path = arguments.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(arguments).items()
    }
    config["fecha_ejecucion"] = datetime.now().isoformat(timespec="seconds")
    (output_dir / "configuracion.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fold_info, global_baseline = describe_folds(
        CSV_PATH, arguments.n_splits, arguments.random_state
    )
    print("Composición de los folds:")
    print(fold_info.drop(columns="baseline_mae").to_string(index=False))

    results = run_cross_validation(
        csv_path=CSV_PATH,
        images_dir=IMAGES_DIR,
        output_dir=output_dir,
        n_splits=arguments.n_splits,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
        optimizer_name=arguments.optimizer,
        learning_rate=arguments.learning_rate,
        backbone_learning_rate=arguments.backbone_learning_rate,
        unfreeze_last_blocks=arguments.unfreeze_last_blocks,
        random_state=arguments.random_state,
        pretrained=not arguments.no_pretrained,
    )

    summary = save_fold_summary(results, fold_info, global_baseline, output_dir)

    print("\nResumen por fold (MAE y RMSE en puntos porcentuales):")
    columns = ["fold", "imagenes_validacion", "mae", "rmse", "r2", "baseline_mae"]
    print(
        summary[columns].to_string(
            index=False, float_format=lambda value: f"{value:.2f}"
        )
    )
    print(
        "\nResultado principal: fila 'global', calculada con las predicciones "
        "de todos los folds juntas."
    )
    print(f"Resultados guardados en: {output_dir}")


if __name__ == "__main__":
    main()
