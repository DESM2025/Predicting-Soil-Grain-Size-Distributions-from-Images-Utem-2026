from pathlib import Path
from typing import Tuple

import pandas as pd
from torch.utils.data import DataLoader

from .granulometria_dataset import GranulometriaDataset
from .splits import split_by_sample_id


PROJECT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_DIR / "datos" / "dataset_final_aumentado.csv"
IMAGES_DIR = PROJECT_DIR / "datos" / "datos_procesados" / "training"


def create_train_val_loaders(
    csv_path: str | Path = CSV_PATH,
    images_dir: str | Path = IMAGES_DIR,
    batch_size: int = 8,
    validation_fraction: float = 0.2,
    random_state: int = 42,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    csv_path = Path(csv_path)
    images_dir = Path(images_dir)

    if not csv_path.is_file():
        raise FileNotFoundError(f"No se encontró el CSV: {csv_path}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"No se encontró la carpeta de imágenes: {images_dir}")
    if batch_size < 1:
        raise ValueError("batch_size debe ser mayor que cero.")
    if num_workers < 0:
        raise ValueError("num_workers no puede ser negativo.")

    dataframe = pd.read_csv(csv_path)
    train_df, validation_df = split_by_sample_id(
        dataframe,
        validation_fraction=validation_fraction,
        random_state=random_state,
    )

    train_dataset = GranulometriaDataset(train_df, images_dir)
    validation_dataset = GranulometriaDataset(validation_df, images_dir)

    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_options,
    )

    return train_loader, validation_loader
