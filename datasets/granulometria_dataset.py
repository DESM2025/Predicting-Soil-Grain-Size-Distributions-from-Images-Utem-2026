from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


TARGET_COLUMNS = [
    "0.002",
    "0.0063",
    "0.02",
    "0.063",
    "0.2",
    "0.63",
    "2",
    "6.3",
    "20",
    "63",
    "200",
]


class ResizeWithPadding:
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        return ImageOps.pad(
            image,
            (self.size, self.size),
            method=Image.Resampling.BICUBIC,
            color=(0, 0, 0),
        )


def default_transform(image_size: int = 512) -> transforms.Compose:
    return transforms.Compose(
        [
            ResizeWithPadding(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class GranulometriaDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        images_dir: str | Path,
        transform: Callable[[Image.Image], Tensor] | None = None,
    ) -> None:
        missing_columns = {"archivo", *TARGET_COLUMNS} - set(dataframe.columns)
        if missing_columns:
            raise ValueError(
                "Faltan columnas requeridas en el DataFrame: "
                + ", ".join(sorted(missing_columns))
            )

        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.images_dir = Path(images_dir)
        self.transform = transform or default_transform()

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        row = self.dataframe.iloc[index]
        image_path = self.images_dir / str(row["archivo"])

        if not image_path.is_file():
            raise FileNotFoundError(f"No se encontró la imagen: {image_path}")

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_tensor = self.transform(image)

        target = torch.tensor(
            row[TARGET_COLUMNS].to_numpy(dtype="float32"),
            dtype=torch.float32,
        )
        return image_tensor, target
