# 4) Data augmentation y preprocesamiento

## Que se hizo

Se creo el script `scripts/augmentation.py`. Por cada una de las 127 fotos de training se guardo la foto original y 5 variantes. Cada variante aplica al azar:

- Volteo horizontal (probabilidad 50 %) y volteo vertical (probabilidad 50 %).
- Rotacion de 0, 90, 180 o 270 grados.
- Brillo y contraste entre 90 % y 110 % del original.
- Ruido gaussiano leve (desviacion 4.0) con probabilidad 30 %.

Los cambios se aplican al azar, pero con semilla fija, por lo que el resultado es reproducible. Todas las variantes conservan la curva granulometrica de su muestra. El `sample_id` de cada foto se obtiene de su nombre de archivo.

Despues, todas las imagenes se pasaron por el redimensionamiento del avance 3. Las fotos de training aumentadas y las fotos de test se procesaron por separado. Las fotos de test no se aumentan.

Flujo de training: fotos originales, data augmentation (762 imagenes), redimensionamiento, carpeta `datos/datos_procesados/training/`.

Flujo de test: fotos originales, redimensionamiento, carpeta `datos/datos_procesados/test/`.

## Resultados

- 127 x 6 = 762 imagenes de training en `datos/datos_procesados/training/`.
- 35 imagenes de test en `datos/datos_procesados/test/`.
- Archivo `datos/dataset_final_aumentado.csv` con 762 filas: nombre del archivo, `sample_id` y las 11 columnas de la curva.
- Las fotos originales no se modificaron.

## Muestras, fotos e imagenes

Los datos de training tienen tres niveles:

- 24 muestras de suelo. Cada muestra (`sample_id`) tiene una sola curva granulometrica medida en laboratorio.
- 127 fotos originales. Cada muestra se fotografio entre 3 y 8 veces (5.3 en promedio), con distintos celulares y encuadres.
- 762 imagenes. Cada foto se multiplica por 6 con el data augmentation (la original y 5 variantes).

Ejemplo: la muestra H038 tiene 8 fotos, que generan 8 x 6 = 48 imagenes, todas con la misma curva.

Por lo tanto, el modelo se entrena con 762 imagenes, pero solo hay 24 curvas distintas que aprender a predecir.

## Observaciones

- El script usa semilla fija 42 (para `random` y `numpy`) y procesa las fotos en orden alfabetico. Asi, cada vez que se ejecuta genera exactamente las mismas variantes.
- La semilla se agrego despues de generar el dataset usado en los entrenamientos. Las imagenes actuales se generaron sin semilla, por lo que una nueva ejecucion producira variantes distintas a las actuales.
- Las 6 imagenes de una misma foto tienen la misma etiqueta. Por eso deben quedar siempre en el mismo conjunto, ya sea training o validacion (ver avance 5).
