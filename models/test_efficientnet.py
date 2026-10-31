import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
from torch import nn

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from datasets import create_train_val_loaders
from models import EfficientNetGranulometria


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _ = create_train_val_loaders(
        batch_size=8,
        num_workers=2,
    )

    images, targets = next(iter(train_loader))
    images = images.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)

    model = EfficientNetGranulometria(
        output_features=11,
        pretrained=True,
        freeze_backbone=True,
    ).to(device)

    predictions = model(images)
    loss = nn.MSELoss()(predictions, targets)
    loss.backward()

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("Prueba de EfficientNet-B0 completada correctamente.")
    print(f"Dispositivo: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    print(f"Entrada: {tuple(images.shape)}")
    print(f"Salida: {tuple(predictions.shape)}")
    print(f"Pérdida MSE: {loss.item():.4f}")
    print(f"Parámetros entrenables: {trainable_parameters:,}")
    print("Backward ejecutado correctamente.")


if __name__ == "__main__":
    main()
