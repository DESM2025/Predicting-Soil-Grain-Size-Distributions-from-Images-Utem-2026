# Predicción de granulometría del suelo mediante imágenes

El objetivo de este proyecto es desarrollar un sistema predictivo de visión
computacional y modelado de datos para estimar con precisión curvas de
granulometría a partir de imágenes de muestras de suelo.

Se explorará el uso de técnicas de deep learning, transfer learning y
proyección geométrica 2D a 3D para extraer información visual relevante y
generar las predicciones requeridas. El proyecto forma parte de la competencia de Kaggle
[Predicting Soil Grain Size Distributions from Images](https://www.kaggle.com/competitions/soil-grain-size-from-photos/overview).

## Información del proyecto

- **Asignatura:** Gestión de Proyectos en Ciencia de Datos
- **Institución:** Universidad Tecnológica Metropolitana
- **Profesor:** Felipe Ernesto Espinoza Vidal

### Equipo

- Diego Ernesto Silva Madariaga  
- Felipe Martínez Gonzáles
- Christian Pérez Flores
- Benjamín Zamorano Soto

## Datos y preprocesamiento

El conjunto disponible contiene imágenes de entrenamiento, imágenes de prueba y archivos CSV con las etiquetas granulométricas. El flujo actual incluye:

1. Análisis exploratorio de las curvas granulométricas.
2. Data augmentation para ampliar el conjunto de entrenamiento.
3. Redimensionamiento de las imágenes a un máximo de 1024 píxeles.
4. Separación de las imágenes procesadas de training y test.

El conjunto de entrenamiento aumentado contiene 762 imágenes y sus etiquetas se encuentran en `datos/dataset_final_aumentado.csv`. Las imágenes procesadas se organizan en:

```text
datos/datos_procesados/training/
datos/datos_procesados/test/
```

## Preparación para el modelo

El pipeline de datos ya cuenta con:

- `datasets/splits.py`: división de entrenamiento y validación agrupada por
  `sample_id`, evitando separar las variantes de una misma muestra.
- `datasets/granulometria_dataset.py`: clase `GranulometriaDataset` para cargar
  imágenes RGB de `512 × 512` y sus 11 valores granulométricos.
- `datasets/dataloaders.py`: creación de los `DataLoader` de entrenamiento y
  validación.

La configuración validada actualmente es:

```text
batch_size=8
num_workers=2
pin_memory=True
GPU: NVIDIA GeForce RTX 4060
```

La prueba del pipeline se ejecuta con:

```powershell
python .\datasets\test_dataloaders.py
```

El modelo seleccionado para la siguiente etapa es EfficientNet-B0
preentrenada mediante transfer learning. Todavía no se ha implementado el
entrenamiento del modelo.

## Instalación del entorno

Se recomienda utilizar un entorno Conda independiente para evitar conflictos
entre las librerías científicas de Anaconda y PyTorch. Desde la raíz del
proyecto, ejecutar:

```powershell
conda create -n utem python=3.14 nomkl -y
conda activate utem
python -m pip install -r requirements.txt
```

El archivo `requirements.txt` instala PyTorch con CUDA 12.8 para utilizar una
GPU NVIDIA compatible. Para comprobar que el entorno, los `DataLoaders` y la
GPU funcionan correctamente:

```powershell
python .\datasets\test_dataloaders.py
```

El resultado debe indicar que el dispositivo utilizado es `cuda` y mostrar la
GPU NVIDIA detectada.

Para trabajar posteriormente en el proyecto, activar siempre el entorno:

```powershell
conda activate utem
```

No se recomienda instalar las dependencias directamente en el entorno
`base`.

## Ejecución

Scripts principales:

```powershell
python .\EDA\eda.py
python .\scripts\augmentation.py
python .\scripts\redimensionar_imagenes.py
```

Las credenciales de Kaggle se configuran localmente mediante `.env`. Este archivo no debe subirse al repositorio.
