### Instalación del modelo y prueba de transfer learning

Se configuró el entorno Conda utem  con las dependencias necesarias para utilizar PyTorch y torchvision con soporte CUDA 12.8. También se incorporó nomkl para evitar conflictos entre las librerías científicas de Anaconda y PyTorch. La GPU NVIDIA GeForce RTX 4060 fue reconocida correctamente por PyTorch.

Posteriormente, se implementó EfficientNet-B0 con pesos preentrenados de ImageNet. Se reemplazó la capa final para generar 11 valores correspondientes a la curva granulométrica y se congeló inicialmente el backbone para aplicar transfer learning.

La integración se probó con un batch real del dataset, ejecutando una predicción, el cálculo de la pérdida MSE y el algoritmo backward, sin realizar todavía un entrenamiento completo.

Resultados:

- GPU utilizada: NVIDIA GeForce RTX 4060.
- Entrada: `(8, 3, 512, 512)`.
- Salida: `(8, 11)`.
- Pérdida MSE calculada correctamente.
- `Backward` ejecutado correctamente.
- Pesos preentrenados descargados y cargados sin errores.

Prueba de EfficientNet-B0 completada correctamente.
- Dispositivo: cuda
- GPU: NVIDIA GeForce RTX 4060
- Entrada: (8, 3, 512, 512)
- Salida: (8, 11)
- Pérdida MSE: 5072.3462
- Parámetros entrenables: 14,091
- Backward ejecutado correctamente.

Ambos sprints fueron completados. El entrenamiento con varias épocas, actualización de pesos, validación y fine-tuning queda como la siguiente etapa.
