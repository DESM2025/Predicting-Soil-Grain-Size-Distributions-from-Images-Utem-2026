
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from datasets.granulometria_dataset import (
    TARGET_COLUMNS,
    GranulometriaDataset,
)
from datasets.splits import split_by_sample_id
from models.efficientnet_model import EfficientNetGranulometria
from training.postprocesamiento import (
    postprocesar_curva,
    validar_curva,
)
from training.train import (
    CSV_PATH,
    IMAGES_DIR,
    OUTPUT_DIR,
    collect_predictions,
    create_loader,
)


def main():
    salida = OUTPUT_DIR / "postprocesamiento"
    salida.mkdir(parents=True, exist_ok=True)

    checkpoint_path = OUTPUT_DIR / "modelo_final.pt"

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"No se encontro el modelo: {checkpoint_path}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Dispositivo:", device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model = EfficientNetGranulometria(
        pretrained=False,
        freeze_backbone=True,
        unfreeze_last_blocks=checkpoint.get(
            "unfreeze_last_blocks", 1
        ),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataframe = pd.read_csv(CSV_PATH)

    _, validation_dataframe = split_by_sample_id(
        dataframe,
        validation_fraction=0.2,
        random_state=42,
    )

    loader = create_loader(
        validation_dataframe,
        IMAGES_DIR,
        batch_size=8,
        num_workers=0,
        shuffle=False,
    )

    targets, predictions = collect_predictions(
        model, loader, device
    )

    reales = targets.numpy()
    originales = predictions.numpy()
    corregidas = postprocesar_curva(originales)

    validas = all(
        validar_curva(curva) for curva in corregidas
    )

    print("\nRESULTADOS DEL POST-PROCESAMIENTO")
    print("Curvas procesadas:", len(corregidas))
    print("Curvas validas:", validas)

    print(
        "Valores fuera de rango antes:",
        int(np.sum(
            (originales < 0) | (originales > 100)
        )),
    )

    print(
        "Descensos antes:",
        int(np.sum(np.diff(originales, axis=1) < 0)),
    )

    print(
        "Valores fuera de rango despues:",
        int(np.sum(
            (corregidas < 0) | (corregidas > 100)
        )),
    )

    print(
        "Descensos despues:",
        int(np.sum(np.diff(corregidas, axis=1) < 0)),
    )

    print(
        "Ultimo tamiz igual a 100%:",
        bool(np.all(corregidas[:, -1] == 100)),
    )

    # Guardar predicciones originales y corregidas
    columnas_originales = [
        f"original_{col}" for col in TARGET_COLUMNS
    ]
    columnas_corregidas = [
        f"corregida_{col}" for col in TARGET_COLUMNS
    ]

    resultados = pd.DataFrame(
        np.column_stack([originales, corregidas]),
        columns=columnas_originales + columnas_corregidas,
    )
    resultados.to_csv(
        salida / "predicciones_postprocesadas.csv",
        index=False,
    )

    # Elegir una curva con cambios visibles
    diferencias = np.sum(
        np.abs(corregidas - originales), axis=1
    )
    indice = int(np.argmax(diferencias))

    x = np.arange(len(TARGET_COLUMNS))

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(
        x, reales[indice], "o-",
        label="Curva real", linewidth=2,
    )
    ax.plot(
        x, originales[indice], "o--",
        label="Prediccion original",
    )
    ax.plot(
        x, corregidas[indice], "o-",
        label="Prediccion corregida",
        linewidth=2,
    )

    ax.set_xticks(x, TARGET_COLUMNS, rotation=45)
    ax.set_xlabel("Tamano de particula (mm)")
    ax.set_ylabel("Porcentaje que pasa (%)")
    ax.set_title(
        "Validacion del post-procesamiento "
        "- EfficientNet-B0"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        salida / "comparacion_curvas_reales.png",
        dpi=160,
    )
    plt.close(fig)

    print("\nArchivos generados en:", salida)
    print("Prueba finalizada correctamente.")


if __name__ == "__main__":
    main()
