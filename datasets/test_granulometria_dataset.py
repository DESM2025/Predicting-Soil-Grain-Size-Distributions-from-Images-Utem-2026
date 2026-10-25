import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from datasets import GranulometriaDataset, split_by_sample_id


CSV_PATH = PROJECT_DIR / "datos" / "dataset_final_aumentado.csv"
IMAGES_DIR = PROJECT_DIR / "datos" / "datos_procesados" / "training"


def main() -> None:
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"No se encontró el CSV: {CSV_PATH}")
    if not IMAGES_DIR.is_dir():
        raise FileNotFoundError(
            f"No se encontró la carpeta de imágenes: {IMAGES_DIR}"
        )

    dataframe = pd.read_csv(CSV_PATH)
    train_df, validation_df = split_by_sample_id(
        dataframe,
        validation_fraction=0.2,
        random_state=42,
    )

    train_ids = set(train_df["sample_id"])
    validation_ids = set(validation_df["sample_id"])
    shared_ids = train_ids & validation_ids
    if shared_ids:
        raise AssertionError(
            f"Hay sample_id compartidos entre train y validación: {shared_ids}"
        )

    for subset_name, subset in (
        ("training", train_df),
        ("validación", validation_df),
    ):
        missing_paths = [
            str(IMAGES_DIR / filename)
            for filename in subset["archivo"]
            if not (IMAGES_DIR / str(filename)).is_file()
        ]
        if missing_paths:
            raise FileNotFoundError(
                f"Faltan {len(missing_paths)} imágenes en {subset_name}. "
                f"Primera ruta: {missing_paths[0]}"
            )

    train_dataset = GranulometriaDataset(train_df, IMAGES_DIR)
    validation_dataset = GranulometriaDataset(validation_df, IMAGES_DIR)

    train_image, train_target = train_dataset[0]
    validation_image, validation_target = validation_dataset[0]

    expected_image_shape = (3, 512, 512)
    expected_target_shape = (11,)
    for subset_name, image, target in (
        ("training", train_image, train_target),
        ("validación", validation_image, validation_target),
    ):
        if tuple(image.shape) != expected_image_shape:
            raise AssertionError(
                f"Forma de imagen inválida en {subset_name}: {tuple(image.shape)}"
            )
        if tuple(target.shape) != expected_target_shape:
            raise AssertionError(
                f"Forma de etiqueta inválida en {subset_name}: {tuple(target.shape)}"
            )

    loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
    images, targets = next(iter(loader))
    if tuple(images.shape) != (2, 3, 512, 512):
        raise AssertionError(f"Forma de batch inválida: {tuple(images.shape)}")
    if tuple(targets.shape) != (2, 11):
        raise AssertionError(f"Forma de etiquetas inválida: {tuple(targets.shape)}")
    if not torch.isfinite(images).all() or not torch.isfinite(targets).all():
        raise AssertionError("El batch contiene valores no finitos.")

    print("Prueba del Dataset completada correctamente.")
    print(f"Filas training/validación: {len(train_df)}/{len(validation_df)}")
    print(f"IDs training/validación: {len(train_ids)}/{len(validation_ids)}")
    print(f"IDs compartidos: {len(shared_ids)}")
    print(f"Imagen individual: {tuple(train_image.shape)}")
    print(f"Etiqueta individual: {tuple(train_target.shape)}")
    print(f"Batch de imágenes: {tuple(images.shape)}")
    print(f"Batch de etiquetas: {tuple(targets.shape)}")


if __name__ == "__main__":
    main()
