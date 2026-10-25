import sys
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from datasets import create_train_val_loaders


def main() -> None:
    train_loader, validation_loader = create_train_val_loaders(
        batch_size=8,
        num_workers=2,
    )

    train_images, train_targets = next(iter(train_loader))
    validation_images, validation_targets = next(iter(validation_loader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_images = train_images.to(device, non_blocking=True)
    train_targets = train_targets.to(device, non_blocking=True)
    validation_images = validation_images.to(device, non_blocking=True)
    validation_targets = validation_targets.to(device, non_blocking=True)

    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(0)
    else:
        device_name = "CPU"

    print("Prueba de DataLoaders completada correctamente.")
    print(f"Dispositivo: {device} ({device_name})")
    print(f"Batches de training: {len(train_loader)}")
    print(f"Batches de validación: {len(validation_loader)}")
    print(f"Training - imágenes: {tuple(train_images.shape)}")
    print(f"Training - etiquetas: {tuple(train_targets.shape)}")
    print(f"Validación - imágenes: {tuple(validation_images.shape)}")
    print(f"Validación - etiquetas: {tuple(validation_targets.shape)}")


if __name__ == "__main__":
    main()
