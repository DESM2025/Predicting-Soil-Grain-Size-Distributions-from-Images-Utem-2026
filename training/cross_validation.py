from dataclasses import dataclass
import random
from typing import Iterator

import pandas as pd


@dataclass(frozen=True)
class GroupedFold:
    """Un fold con grupos completos de muestras."""

    fold_number: int
    train: pd.DataFrame
    validation: pd.DataFrame
    validation_ids: tuple[str, ...]


def create_grouped_folds(
    dataframe: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> list[GroupedFold]:
    """Crea folds sin separar las variantes de un mismo sample_id."""
    required_columns = {"sample_id", "archivo"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(
            "Faltan columnas requeridas para validación cruzada: "
            + ", ".join(sorted(missing_columns))
        )
    if n_splits < 2:
        raise ValueError("n_splits debe ser al menos 2.")

    if dataframe["sample_id"].isna().any():
        raise ValueError("Todos los registros deben tener un sample_id válido.")
    groups = dataframe["sample_id"].astype(str).str.strip()
    if (groups == "").any():
        raise ValueError("Todos los registros deben tener un sample_id válido.")

    unique_ids = list(dict.fromkeys(groups.tolist()))
    if len(unique_ids) < n_splits:
        raise ValueError(
            f"Se necesitan al menos {n_splits} sample_id para crear los folds; "
            f"se encontraron {len(unique_ids)}."
        )

    random_generator = random.Random(random_state)
    random_generator.shuffle(unique_ids)
    fold_ids = [unique_ids[index::n_splits] for index in range(n_splits)]
    folds: list[GroupedFold] = []

    for fold_index, validation_ids in enumerate(fold_ids, start=1):
        validation_id_set = set(validation_ids)
        validation_mask = groups.isin(validation_id_set)
        train_df = dataframe.loc[~validation_mask].reset_index(drop=True)
        validation_df = dataframe.loc[validation_mask].reset_index(drop=True)

        if train_df.empty or validation_df.empty:
            raise ValueError(f"El fold {fold_index} produjo un conjunto vacío.")

        folds.append(
            GroupedFold(
                fold_number=fold_index,
                train=train_df,
                validation=validation_df,
                validation_ids=tuple(validation_ids),
            )
        )

    return folds


def iter_grouped_folds(
    dataframe: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> Iterator[GroupedFold]:
    """Itera sobre folds agrupados manteniendo una API sencilla."""
    yield from create_grouped_folds(dataframe, n_splits, random_state)
