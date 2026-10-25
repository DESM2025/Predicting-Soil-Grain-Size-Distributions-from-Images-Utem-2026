### Configuración de DataLoaders

Se implementó una función para crear los DataLoader de entrenamiento y validación a partir del dataset previamente dividido por sample_id. Esta configuración permite entregar los datos al futuro modelo en batches, manteniendo separados ambos conjuntos.

Se estableció un batch_size de 8 imágenes, con una resolución de entrada de 512 × 512 píxeles y 11 valores granulométricos por muestra. Para acelerar la carga de datos se configuraron 2 workers, pin_memory=True, shuffle=True para entrenamiento y shuffle=False para validación.

La configuración fue probada correctamente utilizando la GPU NVIDIA GeForce RTX 4060. Se generaron 78 batches de entrenamiento y 18 de validación, con las siguientes dimensiones:

- Imágenes: (8, 3, 512, 512).
- Etiquetas: (8, 11).

El tamaño del batch podrá aumentarse posteriormente. Las épocas, tasa de aprendizaje, el optimizador y la función de pérdida se definirán en la etapa de entrenamiento del modelo EfficientNet-B0.
