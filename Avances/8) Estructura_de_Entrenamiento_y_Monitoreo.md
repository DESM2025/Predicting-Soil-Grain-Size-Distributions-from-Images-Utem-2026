### Entrenamiento, validación cruzada y monitoreo

Se creó la carpeta `training/` para separar el entrenamiento de la preparación
de datos y de la definición del modelo. La implementación contempla dos
etapas: primero, la validación cruzada para comparar configuraciones; después,
el entrenamiento final de un único modelo con todos los datos de training.

La implementación incluye:

- Validación cruzada agrupada por `sample_id`, evitando que las variantes de
  una misma muestra aparezcan en training y validación.
- Entrenamiento por épocas con EfficientNet-B0.
- Funciones separadas para entrenamiento y validación de cada época.
- Registro de `train_loss`, `validation_loss` y `learning_rate`.
- Exportación del historial a CSV.
- Gráficos formales de pérdida con una paleta azul sobria.
- MAE, RMSE, MSE y R² para cada fold y para el resumen global.
- MAE por tamaño de partícula.
- Gráficos de valores reales contra predichos y curvas granulométricas.
- Guardado del mejor checkpoint de cada fold.
- Entrenamiento final de un único modelo con la configuración seleccionada.
- Guardado de `modelo_final.pt` y del historial del entrenamiento final.

La configuración inicial utilizada como baseline fue:

- Optimizador: AdamW.
- Learning rate: `1e-3`.
- Weight decay: `1e-4`.
- Batch size: `8`.
- Épocas: `10`.
- Backbone de EfficientNet-B0 congelado inicialmente.
- Validación cruzada: `5` folds.

El baseline obtuvo:

- MSE: `600.60`.
- MAE: `18.02` puntos porcentuales.
- RMSE: `24.51` puntos porcentuales.
- R²: `-0.2879`.

El R² negativo indicó que el modelo todavía era inferior a una referencia que
predice el promedio de cada salida.

Para ejecutar una primera validación cruzada:

```powershell
python -m training.train
```

Los checkpoints, historiales y gráficos se guardan en
`resultados/entrenamiento/`, una carpeta excluida del repositorio.

La pérdida MSE se conserva para optimizar el modelo, pero el resultado se
interpreta principalmente mediante MAE y RMSE, expresados en puntos
porcentuales. La matriz de confusión no aplica directamente porque el problema
es una regresión multisalida, no una clasificación.

## Comparación de configuraciones

Se evaluó una segunda configuración manteniendo el backbone congelado y usando
AdamW con `learning_rate=1e-4` durante 20 épocas:

```powershell
python -m training.train --learning-rate 0.0001 --epochs 20 `
  --experiment-name adamw_lr1e-4_20epocas
```

El resultado fue inferior al baseline:

- MSE: `2545.19`.
- MAE: `38.78` puntos porcentuales.
- RMSE: `50.45` puntos porcentuales.
- R²: `-4.4577`.

Las curvas mostraron que el modelo todavía estaba subentrenado al llegar a la
época 20. Por lo tanto, `1e-4` no fue seleccionado para la siguiente etapa con
el backbone congelado.

Luego se evaluó AdamW con `learning_rate=1e-3` durante 20 épocas:

```powershell
python -m training.train --learning-rate 0.001 --epochs 20 `
  --experiment-name adamw_lr1e-3_20epocas
```

Esta configuración obtuvo el mejor resultado hasta el momento:

- MSE: `420.10`.
- MAE: `14.49` puntos porcentuales.
- RMSE: `20.50` puntos porcentuales.
- R²: `0.0992`.

La comparación muestra que aumentar de 10 a 20 épocas manteniendo
`learning_rate=1e-3` redujo el MAE de `18.02` a `14.49` puntos porcentuales y
mejoró el R² desde `-0.2879` hasta `0.0992`. El modelo pasó a superar la
referencia basada en promedios.

La configuración recomendada actualmente es:

- EfficientNet-B0 con backbone congelado.
- Optimizador AdamW.
- Learning rate: `1e-3`.
- Weight decay: `1e-4`.
- Batch size: `8`.
- Épocas: `20`.
- Validación cruzada agrupada: `5` folds.

Esta configuración es la mejor línea base disponible, pero todavía no es el
entrenamiento final. El siguiente paso será evaluar fine-tuning parcial de las
últimas capas de EfficientNet-B0 con un learning rate menor para el backbone.

La implementación permite realizar ese experimento en la misma carpeta de
código, sin alterar la configuración anterior. Por defecto,
`--unfreeze-last-blocks=0` mantiene el backbone congelado. Para el primer
experimento de fine-tuning parcial se puede ejecutar:

```powershell
python -m training.train --learning-rate 0.001 --backbone-learning-rate 0.00001 `
  --unfreeze-last-blocks 1 --epochs 20 `
  --experiment-name finetuning_last_block_lr1e-5
```

En esta configuración la nueva capa final utiliza `1e-3` y el último bloque
descongelado utiliza `1e-5`. Los resultados se guardan en una subcarpeta
independiente para conservar la comparación con la línea base.

El fine-tuning parcial mejoró el rendimiento global respecto a la extracción de
características:

| Métrica | Feature extraction | Fine-tuning parcial |
|---|---:|---:|
| MSE | `420.10` | `384.56` |
| MAE | `14.49` | `13.79` |
| RMSE | `20.50` | `19.61` |
| R² | `0.0992` | `0.1754` |

El MAE disminuyó de `14.49` a `13.79` puntos porcentuales y el RMSE bajó de
`20.50` a `19.61` puntos. Esto indica que el modelo redujo tanto el error
promedio como la influencia de errores grandes. El R² aumentó de `0.0992` a
`0.1754`, por lo que el modelo explica aproximadamente el 17.5 % de la
variabilidad observada y supera con mayor margen la referencia basada en
predecir los promedios.

La mejora fue especialmente visible en las granulometrías medianas y grandes,
desde `0.63 mm` hasta `200 mm`. En esas salidas el MAE disminuyó
aproximadamente entre 1 y 1.6 puntos porcentuales. Los tamaños pequeños se
mantuvieron prácticamente estables, con errores bajos, mientras que las
granulometrías de `0.063 mm` y `0.2 mm` presentaron cambios menores.

Por lo tanto, el fine-tuning parcial del último bloque fue beneficioso y se
establece como la configuración seleccionada para el entrenamiento final.

## Entrenamiento final de un solo modelo

El comando principal dejó de ejecutar validación cruzada para el entrenamiento
final. Ahora entrena un único modelo con una división agrupada de 80 % para
training y 20 % para validación, utilizando `sample_id` para evitar que las
variantes de una misma muestra queden en conjuntos distintos. Usa la
configuración seleccionada:

- Último bloque de EfficientNet-B0 descongelado.
- AdamW.
- Learning rate de la capa final: `1e-3`.
- Learning rate del backbone: `1e-5`.
- Batch size: `8`.
- Épocas: `20`.

La división utilizada contiene 618 registros para training y 144 para
validación, sin `sample_id` compartidos entre ambos conjuntos. De esta forma,
el entrenamiento final permite revisar el comportamiento del modelo en cada
época sin repetir cinco entrenamientos independientes.

Para ejecutarlo:

```powershell
python -m training.train
```

El modelo se guarda como `modelo_final.pt`, junto con
`historial_entrenamiento_final.csv` y
`perdida_entrenamiento_final.png`. Los artefactos generados anteriormente por
los folds se reemplazan en la carpeta principal, mientras que sus historiales
y gráficos se conservan en
`resultados/entrenamiento/monitoreo_validacion_cruzada/`.

Durante cada época se muestran `train_loss` y `val_loss`. Al finalizar, se
calculan MAE, RMSE, MSE y R² sobre el 20 % de validación, y se guardan las
métricas y gráficos correspondientes.

### Resultado del modelo final

El entrenamiento final terminó correctamente en 20 épocas y guardó el mejor
checkpoint en la época 19, que obtuvo una pérdida de validación de `199.3261`.
Las métricas principales de este modelo son:

- MSE: `199.33`.
- MAE: `9.47` puntos porcentuales.
- RMSE: `14.12` puntos porcentuales.
- R²: `-4.2936`.

El MAE indica que el modelo presenta un error absoluto promedio de
aproximadamente 9.47 puntos porcentuales sobre las 11 granulometrías. El RMSE
de 14.12 muestra que existen algunos errores mayores que elevan la magnitud
del error al penalizar las diferencias grandes.

El R² negativo debe interpretarse con cautela en esta división. El conjunto de
validación contiene poca variabilidad en algunas salidas, especialmente en
`63 mm` y `200 mm`, donde los valores reales son muy similares o constantes.
En ese contexto, el R² puede ser muy negativo incluso cuando el MAE y el RMSE
son relativamente bajos. Por esta razón, para este modelo se priorizan el MAE
y el RMSE como medidas directas del error en puntos porcentuales, mientras que
el R² se informa como referencia complementaria.

Este resultado corresponde al modelo final evaluado con una división única
agrupada de 80 % training y 20 % validación. La validación cruzada queda
únicamente como registro histórico de experimentación y no como el resultado
principal del proyecto.
