### Implementación del dataset y división de datos

Se implementó una clase GranulometriaDataset utilizando PyTorch para cargar las imágenes redimensionadas y asociarlas con sus respectivas curvas granulométricas. Las imágenes se convierten a RGB, se ajustan a 512 × 512 píxeles mediante padding para conservar sus proporciones y se normalizan antes de ser entregadas al modelo.

También se implementó una división agrupada por sample_id, evitando que imágenes aumentadas de una misma muestra aparezcan simultáneamente en los conjuntos de entrenamiento y validación. La división utiliza una semilla fija para mantener resultados reproducibles.

La validación del pipeline se realizó mediante un script de prueba, comprobando que las rutas existan, que las imágenes y etiquetas tengan las dimensiones esperadas y que el DataLoader genere batches correctamente.

Los resultados obtenidos fueron:

- 618 filas para entrenamiento y 144 para validación.
- 19 sample_id en entrenamiento y 5 en validación.
- 0 sample_id compartidos entre ambos conjuntos.
- Imágenes individuales con forma (3, 512, 512).
- Etiquetas individuales con forma (11,).
- Batches de imágenes con forma (2, 3, 512, 512).
- Batches de etiquetas con forma (2, 11).

La prueba finalizó correctamente, dejando preparado el pipeline de datos para conectar posteriormente un modelo de transfer learning.

Finalmente, se configuró PyTorch con soporte CUDA y se verificó el reconocimiento de la GPU NVIDIA GeForce RTX 4060. Para la siguiente etapa se seleccionó EfficientNet-B0 preentrenada mediante transfer learning, que utilizará el pipeline de datos ya validado.
