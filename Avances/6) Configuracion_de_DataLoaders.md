# 6) Configuracion de DataLoaders

## Que se hizo

Se creo la funcion `create_train_val_loaders` (`datasets/dataloaders.py`). Usa la division por muestra del avance 5 y entrega los datos al modelo en batches.

Configuracion:

- `batch_size = 8`
- `num_workers = 2`
- `pin_memory = True`
- `shuffle = True` en training y `shuffle = False` en validacion.

Se probo con el script `datasets/test_dataloaders.py` en la GPU NVIDIA GeForce RTX 4060.

## Resultados

- 78 batches de training (618 imagenes) y 18 batches de validacion (144 imagenes).
- Batch de imagenes: (8, 3, 512, 512).
- Batch de etiquetas: (8, 11).
- La prueba termino sin errores.
