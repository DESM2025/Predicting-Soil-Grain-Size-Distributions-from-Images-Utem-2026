# 7) EfficientNet-B0 y transfer learning

## Que se hizo

- Se configuro el entorno Conda `utem` con PyTorch y torchvision con soporte CUDA 12.8. Se agrego `nomkl` para evitar conflictos entre librerias. PyTorch reconocio la GPU NVIDIA GeForce RTX 4060.
- Se creo la clase `EfficientNetGranulometria` (`models/efficientnet_model.py`):
  - Usa EfficientNet-B0 con pesos preentrenados en ImageNet.
  - Se reemplazo la capa final por una de 11 salidas, una por cada tamaño de particula.
  - El resto de la red (backbone) se congela y solo se entrena la capa nueva. Esto es transfer learning.
  - Tiene la opcion `unfreeze_last_blocks` para descongelar los ultimos bloques del backbone (se usa en el avance 8).
- Se probo con `models/test_efficientnet.py` usando un batch real del dataset: prediccion, calculo de la perdida MSE y backward, sin entrenar todavia.

## Resultados

- Entrada (8, 3, 512, 512) y salida (8, 11).
- Perdida MSE inicial: 5072.35 (modelo aun sin entrenar).
- Parametros entrenables: 14,091, que corresponden a la capa final (1280 x 11 pesos + 11 sesgos).
- Los pesos preentrenados se cargaron y el backward se ejecuto sin errores.
