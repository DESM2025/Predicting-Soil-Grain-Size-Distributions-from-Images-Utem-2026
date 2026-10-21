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

## Ejecución

Instalar las dependencias:

```powershell
pip install -r requirements.txt
```

Scripts principales:

```powershell
python .\EDA\eda.py
python .\scripts\augmentation.py
python .\scripts\redimensionar_imagenes.py
```

Las credenciales de Kaggle se configuran localmente mediante `.env`. Este archivo no debe subirse al repositorio.
