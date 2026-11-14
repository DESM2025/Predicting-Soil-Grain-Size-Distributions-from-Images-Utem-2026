# 5) Dataset y division de datos

## Que se hizo

- Clase `GranulometriaDataset` (`datasets/granulometria_dataset.py`). Para cada fila del CSV:
  - Carga la imagen y la convierte a RGB.
  - La ajusta a 512x512 pixeles agregando bordes negros (padding), para no deformarla.
  - La normaliza con los valores de ImageNet, que son los que espera el modelo preentrenado.
  - Entrega la imagen junto con sus 11 valores de curva.
- Funcion `split_by_sample_id` (`datasets/splits.py`). Separa 80 % para training y 20 % para validacion por `sample_id`, con semilla fija 42. Todas las imagenes de una misma muestra (originales y aumentadas) quedan en el mismo conjunto.
- Script de prueba `datasets/test_granulometria_dataset.py`.

## Por que se divide por muestra

Si una foto queda en training y su version rotada queda en validacion, el modelo se evaluaria con algo que practicamente ya vio, y el error saldria artificialmente bajo. Al dividir por muestra, la validacion siempre usa suelos que el modelo no conoce.

## Resultados

- Training: 618 imagenes de 19 muestras.
- Validacion: 144 imagenes de 5 muestras.
- 0 muestras compartidas entre training y validacion.
- Cada imagen tiene forma (3, 512, 512) y cada etiqueta tiene forma (11,).
- Batch de prueba de 2 imagenes: imagenes (2, 3, 512, 512) y etiquetas (2, 11).
