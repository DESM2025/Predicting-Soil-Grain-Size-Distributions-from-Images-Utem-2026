# 1) Analisis exploratorio de datos (EDA)

## Que se hizo

Se analizo el archivo `datos/Training_labels_updated.csv` con el script `EDA/eda.py`. El archivo tiene 24 muestras de suelo (`sample_id`) y 11 columnas con el porcentaje acumulado de material que pasa por cada tamaño de particula: 0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2, 6.3, 20, 63 y 200 mm.

Se reviso:

- Valores faltantes y filas duplicadas.
- Que cada curva este entre 0 y 100, que no disminuya al aumentar el tamaño y que termine en 100 %.
- Estadisticas por tamaño: media, desviacion estandar y percentiles.
- Porcentaje de material entre cada par de tamaños (fracciones por intervalo).
- Correlacion entre tamaños.

## Resultados

- 24 muestras, sin valores faltantes ni filas duplicadas.
- Las 24 curvas son validas: estan entre 0 y 100, son crecientes y terminan en 100 % en 200 mm.
- La mayor diferencia entre muestras esta en los tamaños medios. La desviacion estandar maxima es de 38.18 puntos en 0.63 mm. En 200 mm la desviacion es 0, porque todas las muestras valen 100 %.
- Los intervalos con mas material en promedio son 0.02-0.063 mm (15.7 %), 0.063-0.2 mm (15.5 %), 2-6.3 mm (14.7 %) y 6.3-20 mm (14.0 %).
- Los tamaños cercanos estan correlacionados positivamente, lo esperable en una curva acumulada.

## Conclusion

Las muestras son muy distintas entre si y son pocas (24). Esto aumenta el riesgo de sobreajuste, por lo que el modelo se debe evaluar siempre con muestras que no vio durante el entrenamiento.

Archivos generados en `EDA/graficos/`: `estadisticas_granulometricas.csv`, `fracciones_intervalos.csv` y los graficos del analisis.
