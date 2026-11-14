# 8) Entrenamiento, validacion cruzada y monitoreo

## Que se hizo

Codigo en la carpeta `training/`:

- `train.py`: entrenamiento por epocas, validacion y entrenamiento del modelo final.
- `cross_validation.py`: creacion de los folds de la validacion cruzada.
- `monitoring.py`: calculo de metricas (MSE, MAE, RMSE, R2 y MAE por tamaño), historial y graficos.
- `ejecutar_validacion_cruzada.py`: vuelve a ejecutar la validacion cruzada con la configuracion final.

Configuracion comun a todos los entrenamientos: perdida MSE, optimizador AdamW, weight decay 1e-4 y batch size 8. En cada epoca se registra la perdida de training, la perdida de validacion y el learning rate. Al final se usa el modelo de la epoca con menor perdida de validacion.

## Como se aplico la validacion cruzada

1. Las 24 muestras se dividen en 5 grupos (folds) de 4 o 5 muestras cada uno.
2. Se entrenan 5 modelos. Cada modelo entrena con 4 grupos y se evalua con el grupo que queda fuera.
3. Se rota el grupo de evaluacion hasta que cada grupo se usa una vez para evaluar.
4. Se juntan las predicciones de los 5 modelos (las 762 imagenes) y se calcula el error sobre todas. Ese es el resultado final.

Asi, cada modelo siempre se evalua con muestras que no vio durante su entrenamiento, y todas las muestras se usan una vez para evaluar.

La division es por muestra (`sample_id`), no por imagen. Cada una de las 24 muestras tiene entre 3 y 8 fotos, y cada foto genera 6 imagenes (ver avance 4), por lo que todas las imagenes de una muestra quedan siempre en el mismo grupo. Si se dividiera por imagen, algunas fotos de una muestra quedarian en training y otras en validacion, y el modelo se evaluaria con un suelo que ya conoce. Esta tecnica se llama Group K-Fold. Se uso semilla 42, por lo que todas las configuraciones se evaluaron con los mismos 5 grupos y sus resultados se pueden comparar directamente.

Cantidad de imagenes por fold (6 imagenes por foto: la original y 5 variantes):

| Fold | Imagenes train | Muestras train | Imagenes validacion | Muestras validacion | Muestras en validacion |
|---|---|---|---|---|---|
| 1 | 606 | 19 | 156 | 5 | F827, G190, H183, H367, H493 |
| 2 | 582 | 19 | 180 | 5 | H038, H371, H374, H549, H668 |
| 3 | 600 | 19 | 162 | 5 | H030, H037, H368, H615, H666 |
| 4 | 618 | 19 | 144 | 5 | H126, H405, H616, H617, H637 |
| 5 | 642 | 20 | 120 | 4 | H181, H366, H372, H516 |
| Total | | | 762 | 24 | |


Cada muestra aparece en validacion una sola vez. Sumando los 5 folds, las 762 imagenes (24 muestras) se usan para evaluar, cada una con un modelo que no la vio. La cantidad de imagenes cambia entre folds porque no todas las muestras tienen la misma cantidad de fotos.

## Como se calculan las metricas de la validacion cruzada

- Por fold: cada modelo se evalua solo con su grupo de validacion (`metrics_fold_1.csv` a `metrics_fold_5.csv`).
- Global: se juntan las predicciones de los 5 folds (762 imagenes) y se calculan las metricas sobre todas juntas (`metrics_summary.csv`). Este es el resultado principal.

El promedio simple de los 5 folds da un valor algo distinto al global, porque los folds no tienen la misma cantidad de imagenes. Para la configuracion final, el MAE global es 13.60 y el promedio simple es 13.41.

## Configuraciones comparadas con validacion cruzada

MAE y RMSE en puntos porcentuales.

| Configuracion | Epocas | MAE | RMSE | R2 |
|---|---|---|---|---|
| 1. Backbone congelado, lr 1e-3 | 10 | 18.02 | 24.51 | -0.29 |
| 2. Backbone congelado, lr 1e-4 | 20 | 38.78 | 50.45 | -4.46 |
| 3. Backbone congelado, lr 1e-3 | 20 | 14.49 | 20.50 | 0.10 |
| 4. Ultimo bloque descongelado, lr 1e-3 capa final y 1e-5 backbone | 20 | 13.60 | 19.29 | 0.20 |

- Con lr 1e-4 el modelo aprende demasiado lento: a las 20 epocas todavia no converge.
- Con lr 1e-3, pasar de 10 a 20 epocas baja el MAE de 18.02 a 14.49.
- Descongelar el ultimo bloque del backbone baja el MAE a 13.60. Ademas tiene menor error que las otras 3 configuraciones en los 5 folds.

MAE por fold de cada configuracion:

| Fold | Config. 1 | Config. 2 | Config. 3 | Config. 4 | Baseline |
|---|---|---|---|---|---|
| 1 | 25.86 | 45.30 | 21.30 | 19.57 | 17.35 |
| 2 | 17.02 | 37.36 | 13.97 | 13.21 | 16.07 |
| 3 | 15.90 | 35.61 | 12.17 | 11.56 | 13.34 |
| 4 | 18.10 | 46.02 | 14.81 | 13.87 | 17.49 |
| 5 | 12.52 | 28.05 | 9.19 | 8.82 | 17.89 |
| Global | 18.02 | 38.78 | 14.49 | 13.60 | 16.31 |

Se selecciono la configuracion 4.

La columna Baseline es una referencia: predecir siempre la curva promedio del training, sin mirar la imagen. La configuracion 4 mejora esa referencia en 4 de los 5 folds. En el fold 1 queda peor (19.57 contra 17.35).

El grafico `resultados/entrenamiento/comparacion_configuraciones.png` (generado con `scripts/grafico_comparacion_configuraciones.py`) muestra el MAE global de las 4 configuraciones junto a la referencia.

Nota sobre la configuracion 4: los resultados de esta tabla corresponden a la ejecucion del 14 de noviembre con `python -m training.ejecutar_validacion_cruzada`, guardada en `resultados/entrenamiento/validacion_cruzada/`. La ejecucion original de octubre (carpeta `finetuning_last_block_lr1e-5/`) dio un MAE global de 13.79. La diferencia es pequeña y se debe a la aleatoriedad del entrenamiento en GPU. Las conclusiones no cambian.

## Entrenamiento del modelo final

Con la configuracion 4 se entreno un solo modelo con la division 80/20 por muestra del avance 5, durante 20 epocas:

- Training: 618 imagenes (19 muestras).
- Validacion: 144 imagenes (5 muestras: H183, H368, H374, H516 y H615).

Esta division es una sola particion. No coincide con ninguno de los folds de la validacion cruzada. La mejor epoca fue la 19, con perdida de validacion de 199.33.

Resultados sobre la validacion:

- MSE: 199.33
- MAE: 9.47 puntos porcentuales
- RMSE: 14.12 puntos porcentuales
- R2: -4.29

MAE por tamaño de particula (puntos porcentuales):

| mm | 0.002 | 0.0063 | 0.02 | 0.063 | 0.2 | 0.63 | 2 | 6.3 | 20 | 63 | 200 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MAE | 1.00 | 1.93 | 4.37 | 6.43 | 8.52 | 9.60 | 9.50 | 13.79 | 15.80 | 16.19 | 17.07 |

Interpretacion:

- MAE de 9.47: en promedio, cada valor de la curva predicha se equivoca en 9.5 puntos porcentuales.
- El error aumenta con el tamaño de particula: es bajo en los tamaños finos (entre 1 y 6.4 puntos hasta 0.063 mm) y alto en los gruesos (entre 13.8 y 17.1 puntos desde 6.3 mm).
- El MAE de 9.47 no se debe comparar con el 13.60 de la validacion cruzada. El 9.47 se mide sobre solo 5 muestras de una particion. El 13.60 se mide sobre las 24 muestras y es la estimacion mas confiable del desempeno del modelo.

## Hallazgo sobre el R2

El R2 compara el error del modelo con la variacion que tienen los datos reales con los que se evalua. Si los datos reales varian poco, cualquier error hace que el R2 baje mucho, aunque el error sea chico.

| Evaluacion | Muestras evaluadas | MAE | R2 |
|---|---|---|---|
| Validacion cruzada, global | 24 | 13.60 | 0.20 |
| Validacion cruzada, promedio de los 5 folds | 4 a 5 por fold | 13.41 | -0.92 |
| Fold 5 | 4 | 8.82 | -4.55 |
| Modelo final (80/20) | 5 | 9.47 | -4.29 |

- Con las 24 muestras juntas hay mucha variacion entre curvas, y el R2 es positivo (0.20).
- Con 4 o 5 muestras parecidas entre si hay poca variacion, y el R2 sale muy negativo aunque el MAE sea bajo. El fold 5 tiene el menor MAE de todos (8.82) y aun asi un R2 de -4.55.
- El R2 negativo del modelo final (-4.29) se explica por esto, no porque el modelo sea peor.

Conclusion: con pocas muestras de evaluacion el R2 no es confiable. Se usa el MAE como metrica principal y el R2 solo se reporta con la validacion cruzada global.

## Limitaciones

- Con solo 24 muestras, el error cambia mucho segun que muestras quedan en validacion (MAE entre 8.8 y 19.6 segun el fold).
- La epoca que se guarda se elige con el mismo conjunto de validacion con el que se reportan las metricas. Esto hace que los resultados sean levemente optimistas.
- La salida del modelo no esta restringida: puede entregar valores fuera de 0-100 o curvas que disminuyen. Esto se corrige con el post-procesamiento del Sprint 3.
- La evaluacion final e independiente es el test de Kaggle: 10 muestras que no se usaron en ninguna etapa del entrenamiento ni de la seleccion del modelo.

## Como ejecutar

- Entrenamiento final: `python -m training.train`
- Validacion cruzada con la configuracion 4: `python -m training.ejecutar_validacion_cruzada`
- Validacion cruzada con otra configuracion, por ejemplo la 3: `python -m training.ejecutar_validacion_cruzada --unfreeze-last-blocks 0 --output-dir resultados/entrenamiento/validacion_cruzada_config3`

Las configuraciones 1 a 3 se ejecutaron con una version anterior de `train.py`, que ya no tiene la opcion `--experiment-name`. Hoy se repiten con `ejecutar_validacion_cruzada.py`.

## Archivos de resultados

- `resultados/entrenamiento/monitoreo_validacion_cruzada/`: configuracion 1.
- `resultados/entrenamiento/adamw_lr1e-4_20epocas/`: configuracion 2.
- `resultados/entrenamiento/adamw_lr1e-3_20epocas/`: configuracion 3.
- `resultados/entrenamiento/validacion_cruzada/`: configuracion 4, ejecucion del 14 de noviembre (incluye `resumen_por_fold.csv` y `configuracion.json`).
- `resultados/entrenamiento/finetuning_last_block_lr1e-5/`: configuracion 4, ejecucion original de octubre.
- `resultados/entrenamiento/`: modelo final (`modelo_final.pt`, `historial_entrenamiento_final.csv`, `metricas_validacion_final.csv` y graficos) y `comparacion_configuraciones.png`.

Cada carpeta de configuracion contiene las metricas, el historial y el grafico de perdida de cada fold, el resumen global (`metrics_summary.csv`) y los graficos de valores reales contra predichos, MAE por tamaño y curvas granulometricas.
