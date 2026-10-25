from typing import Tuple

import pandas as pd


def split_by_sample_id(
    dataframe: pd.DataFrame,
    validation_fraction: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required_columns = {"sample_id", "archivo"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(
            "Faltan columnas requeridas para dividir el dataset: "
            + ", ".join(sorted(missing_columns))
        )

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction debe estar entre 0 y 1.")

    if dataframe["sample_id"].isna().any():
        raise ValueError("Todos los registros deben tener un sample_id válido.")
    sample_ids = dataframe["sample_id"].astype(str).str.strip()
    if (sample_ids == "").any():
        raise ValueError("Todos los registros deben tener un sample_id válido.")

    unique_ids = sample_ids.drop_duplicates().tolist()
    if len(unique_ids) < 2:
        raise ValueError("Se necesitan al menos dos sample_id para dividir.")

    generator = pd.Series(unique_ids).sample(
        frac=1.0,
        random_state=random_state,
    )
    validation_count = max(1, round(len(unique_ids) * validation_fraction))
    validation_ids = set(generator.iloc[:validation_count])

    validation_mask = sample_ids.isin(validation_ids)
    train_df = dataframe.loc[~validation_mask].reset_index(drop=True)
    validation_df = dataframe.loc[validation_mask].reset_index(drop=True)

    if train_df.empty or validation_df.empty:
        raise ValueError("La división produjo un conjunto vacío.")

    return train_df, validation_df
