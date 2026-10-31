from argparse import ArgumentParser, Namespace
from pathlib import Path
import shutil
from typing import Literal

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.optim import Adam, AdamW, Optimizer
from torch.utils.data import DataLoader

from datasets.granulometria_dataset import TARGET_COLUMNS, GranulometriaDataset
from datasets.splits import split_by_sample_id
from models.efficientnet_model import EfficientNetGranulometria
from training.cross_validation import GroupedFold, create_grouped_folds
from training.monitoring import (
    RegressionMetrics,
    TrainingHistory,
    calculate_regression_metrics,
    plot_curve_examples,
    plot_loss_curves,
    plot_prediction_diagnostics,
    save_history,
    save_metrics,
)


PROJECT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_DIR / "datos" / "dataset_final_aumentado.csv"
IMAGES_DIR = PROJECT_DIR / "datos" / "datos_procesados" / "training"
OUTPUT_DIR = PROJECT_DIR / "resultados" / "entrenamiento"


def create_optimizer(
    model: nn.Module,
    optimizer_name: Literal["adam", "adamw"] = "adamw",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    backbone_learning_rate: float | None = None,
) -> Optimizer:
    if learning_rate <= 0:
        raise ValueError("learning_rate debe ser mayor que cero.")
    if backbone_learning_rate is not None and backbone_learning_rate <= 0:
        raise ValueError("backbone_learning_rate debe ser mayor que cero.")

    head_parameters = [
        parameter
        for parameter in model.backbone.classifier.parameters()
        if parameter.requires_grad
    ]
    backbone_parameters = [
        parameter
        for parameter in model.backbone.features.parameters()
        if parameter.requires_grad
    ]
    parameter_groups = []
    if head_parameters:
        parameter_groups.append({"params": head_parameters, "lr": learning_rate})
    if backbone_parameters:
        parameter_groups.append(
            {
                "params": backbone_parameters,
                "lr": backbone_learning_rate or learning_rate,
            }
        )
    if not parameter_groups:
        raise ValueError("El modelo no tiene parámetros entrenables.")
    if optimizer_name == "adamw":
        return AdamW(parameter_groups, weight_decay=weight_decay)
    if optimizer_name == "adam":
        return Adam(parameter_groups, weight_decay=weight_decay)
    raise ValueError(f"Optimizador no soportado: {optimizer_name}")


def create_loader(
    dataframe: pd.DataFrame,
    images_dir: Path,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    dataset = GranulometriaDataset(dataframe, images_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(images)
        loss = loss_function(predictions, targets)
        loss.backward()
        optimizer.step()
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size

    if total_items == 0:
        raise ValueError("El DataLoader de entrenamiento está vacío.")
    return total_loss / total_items


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_items = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        loss = loss_function(model(images), targets)
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size

    if total_items == 0:
        raise ValueError("El DataLoader de validación está vacío.")
    return total_loss / total_items


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for images, batch_targets in loader:
        images = images.to(device, non_blocking=True)
        predictions.append(model(images).cpu())
        targets.append(batch_targets)
    if not predictions:
        raise ValueError("El DataLoader de validación está vacío.")
    return torch.cat(targets), torch.cat(predictions)


def run_fold(
    fold: GroupedFold,
    images_dir: Path,
    output_dir: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    num_workers: int,
    optimizer_name: Literal["adam", "adamw"],
    learning_rate: float,
    backbone_learning_rate: float | None,
    unfreeze_last_blocks: int,
    pretrained: bool,
) -> tuple[RegressionMetrics, np.ndarray, np.ndarray]:
    model = EfficientNetGranulometria(
        pretrained=pretrained,
        freeze_backbone=True,
        unfreeze_last_blocks=unfreeze_last_blocks,
    ).to(device)
    optimizer = create_optimizer(
        model,
        optimizer_name,
        learning_rate,
        backbone_learning_rate=backbone_learning_rate,
    )
    loss_function = nn.MSELoss()
    train_loader = create_loader(fold.train, images_dir, batch_size, num_workers, True)
    validation_loader = create_loader(
        fold.validation, images_dir, batch_size, num_workers, False
    )
    history = TrainingHistory()
    best_validation_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, loss_function, optimizer, device
        )
        validation_loss = validate_one_epoch(
            model, validation_loader, loss_function, device
        )
        current_learning_rate = optimizer.param_groups[0]["lr"]
        history.add(epoch, train_loss, validation_loss, current_learning_rate)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "validation_loss": validation_loss,
                    "fold": fold.fold_number,
                    "epoch": epoch,
                },
                output_dir / f"best_model_fold_{fold.fold_number}.pt",
            )
        print(
            f"Fold {fold.fold_number} | época {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={validation_loss:.4f}"
        )

    checkpoint = torch.load(
        output_dir / f"best_model_fold_{fold.fold_number}.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    fold_targets, fold_predictions = collect_predictions(model, validation_loader, device)
    metrics = calculate_regression_metrics(
        fold_targets.numpy(),
        fold_predictions.numpy(),
    )
    save_history(history, output_dir / f"history_fold_{fold.fold_number}.csv")
    save_metrics(metrics, output_dir / f"metrics_fold_{fold.fold_number}.csv")
    plot_loss_curves(
        history,
        output_dir / f"loss_fold_{fold.fold_number}.png",
        title=f"Pérdida por época - fold {fold.fold_number}",
    )
    print(
        f"Fold {fold.fold_number} | MAE={metrics.mae:.2f} puntos | "
        f"RMSE={metrics.rmse:.2f} puntos | R²={metrics.r2:.4f}"
    )
    return metrics, fold_targets.numpy(), fold_predictions.numpy()


def run_cross_validation(
    csv_path: Path = CSV_PATH,
    images_dir: Path = IMAGES_DIR,
    output_dir: Path = OUTPUT_DIR,
    n_splits: int = 5,
    epochs: int = 10,
    batch_size: int = 8,
    num_workers: int = 2,
    optimizer_name: Literal["adam", "adamw"] = "adamw",
    learning_rate: float = 1e-3,
    backbone_learning_rate: float | None = None,
    unfreeze_last_blocks: int = 0,
    random_state: int = 42,
    pretrained: bool = True,
) -> list[RegressionMetrics]:
    dataframe = pd.read_csv(csv_path)
    folds = create_grouped_folds(dataframe, n_splits, random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_results = [
        run_fold(
            fold,
            images_dir,
            output_dir,
            device,
            epochs,
            batch_size,
            num_workers,
            optimizer_name,
            learning_rate,
            backbone_learning_rate,
            unfreeze_last_blocks,
            pretrained,
        )
        for fold in folds
    ]
    results = [result[0] for result in fold_results]
    all_targets = np.concatenate([result[1] for result in fold_results])
    all_predictions = np.concatenate([result[2] for result in fold_results])
    summary = calculate_regression_metrics(all_targets, all_predictions)
    save_metrics(summary, output_dir / "metrics_summary.csv")
    plot_prediction_diagnostics(all_targets, all_predictions, output_dir)
    plot_curve_examples(
        all_targets,
        all_predictions,
        output_dir / "curvas_granulometricas.png",
    )
    print("\nResultados de validación cruzada:")
    print(f"MSE: {summary.mse:.2f}")
    print(f"MAE: {summary.mae:.2f} puntos porcentuales")
    print(f"RMSE: {summary.rmse:.2f} puntos porcentuales")
    print(f"R²: {summary.r2:.4f}")
    print("MAE por tamaño de partícula:")
    for target, mae in zip(TARGET_COLUMNS, summary.mae_by_target):
        print(f"  {target} mm: {mae:.2f} puntos")
    return results


def preserve_cross_validation_monitoring(output_dir: Path) -> None:
    monitoring_dir = output_dir / "monitoreo_validacion_cruzada"
    monitoring_dir.mkdir(parents=True, exist_ok=True)
    preserved_names = (
        "history_fold_",
        "loss_fold_",
        "metrics_fold_",
        "metrics_summary.csv",
        "reales_vs_predichos.png",
        "mae_por_tamano.png",
        "curvas_granulometricas.png",
    )
    for path in output_dir.iterdir():
        if path.name == monitoring_dir.name or not any(
            path.name.startswith(prefix) for prefix in preserved_names
        ):
            continue
        destination = monitoring_dir / path.name
        if destination.exists():
            destination.unlink()
        shutil.move(str(path), str(destination))


def remove_previous_training_outputs(output_dir: Path) -> None:
    for path in output_dir.iterdir():
        if path.name == "monitoreo_validacion_cruzada":
            continue
        if path.is_file() and (
            path.name.startswith("best_model_fold_")
            or path.name.startswith("history_fold_")
            or path.name.startswith("loss_fold_")
            or path.name.startswith("metrics_fold_")
            or path.name in {
                "metrics_summary.csv",
                "reales_vs_predichos.png",
                "mae_por_tamano.png",
                "curvas_granulometricas.png",
            }
        ):
            path.unlink()


def run_final_training(
    csv_path: Path = CSV_PATH,
    images_dir: Path = IMAGES_DIR,
    output_dir: Path = OUTPUT_DIR,
    epochs: int = 20,
    batch_size: int = 8,
    num_workers: int = 2,
    optimizer_name: Literal["adam", "adamw"] = "adamw",
    learning_rate: float = 1e-3,
    backbone_learning_rate: float = 1e-5,
    unfreeze_last_blocks: int = 1,
    validation_fraction: float = 0.2,
    random_state: int = 42,
    pretrained: bool = True,
) -> Path:
    dataframe = pd.read_csv(csv_path)
    if dataframe.empty:
        raise ValueError("El CSV de entrenamiento está vacío.")
    if epochs < 1:
        raise ValueError("epochs debe ser mayor que cero.")
    train_dataframe, validation_dataframe = split_by_sample_id(
        dataframe,
        validation_fraction=validation_fraction,
        random_state=random_state,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    preserve_cross_validation_monitoring(output_dir)
    remove_previous_training_outputs(output_dir)

    model = EfficientNetGranulometria(
        pretrained=pretrained,
        freeze_backbone=True,
        unfreeze_last_blocks=unfreeze_last_blocks,
    ).to(device)
    optimizer = create_optimizer(
        model,
        optimizer_name,
        learning_rate,
        backbone_learning_rate=backbone_learning_rate,
    )
    loss_function = nn.MSELoss()
    train_loader = create_loader(
        train_dataframe,
        images_dir,
        batch_size,
        num_workers,
        True,
    )
    validation_loader = create_loader(
        validation_dataframe,
        images_dir,
        batch_size,
        num_workers,
        False,
    )
    history = TrainingHistory()
    best_loss = float("inf")
    checkpoint_path = output_dir / "modelo_final.pt"

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, loss_function, optimizer, device
        )
        validation_loss = validate_one_epoch(
            model,
            validation_loader,
            loss_function,
            device,
        )
        current_learning_rate = optimizer.param_groups[0]["lr"]
        history.add(epoch, train_loss, validation_loss, current_learning_rate)
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "epoch": epoch,
                    "optimizer": optimizer_name,
                    "learning_rate": learning_rate,
                    "backbone_learning_rate": backbone_learning_rate,
                    "unfreeze_last_blocks": unfreeze_last_blocks,
                },
                checkpoint_path,
            )
        print(
            f"Época {epoch}/{epochs} | train_loss={train_loss:.4f} | "
            f"val_loss={validation_loss:.4f}"
        )

    save_history(history, output_dir / "historial_entrenamiento_final.csv")
    plot_loss_curves(
        history,
        output_dir / "perdida_entrenamiento_final.png",
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    validation_targets, validation_predictions = collect_predictions(
        model,
        validation_loader,
        device,
    )
    metrics = calculate_regression_metrics(
        validation_targets.numpy(),
        validation_predictions.numpy(),
    )
    save_metrics(metrics, output_dir / "metricas_validacion_final.csv")
    plot_prediction_diagnostics(
        validation_targets.numpy(),
        validation_predictions.numpy(),
        output_dir,
        title_prefix="Validación final",
    )
    plot_curve_examples(
        validation_targets.numpy(),
        validation_predictions.numpy(),
        output_dir / "curvas_validacion_finales.png",
    )
    print("\nMétricas de validación final:")
    print(f"MAE: {metrics.mae:.2f} puntos porcentuales")
    print(f"RMSE: {metrics.rmse:.2f} puntos porcentuales")
    print(f"R²: {metrics.r2:.4f}")
    print(f"\nModelo final guardado en: {checkpoint_path}")
    return checkpoint_path


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Entrenamiento final de EfficientNet-B0.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Proporción agrupada reservada para validación.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--backbone-learning-rate",
        type=float,
        default=None,
        help="Learning rate opcional para los bloques descongelados.",
    )
    parser.add_argument(
        "--unfreeze-last-blocks",
        type=int,
        default=1,
        help="Cantidad de bloques finales del backbone que se descongelan.",
    )
    parser.add_argument("--optimizer", choices=("adamw", "adam"), default="adamw")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.epochs < 1:
        raise ValueError("--epochs debe ser mayor que cero.")
    if arguments.unfreeze_last_blocks < 0:
        raise ValueError("--unfreeze-last-blocks no puede ser negativo.")
    if arguments.unfreeze_last_blocks and arguments.backbone_learning_rate is None:
        arguments.backbone_learning_rate = 1e-5
    run_final_training(
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
        learning_rate=arguments.learning_rate,
        backbone_learning_rate=arguments.backbone_learning_rate,
        unfreeze_last_blocks=arguments.unfreeze_last_blocks,
        validation_fraction=arguments.validation_fraction,
        optimizer_name=arguments.optimizer,
        output_dir=arguments.output_dir,
        pretrained=not arguments.no_pretrained,
    )


if __name__ == "__main__":
    main()
