# 3) Redimensionamiento de imagenes

## Que se hizo

Se creo el script `scripts/redimensionar_imagenes.py` (libreria Pillow). Para cada imagen:

- Reduce la imagen para que su lado mayor mida como maximo 1024 pixeles.
- Mantiene la proporcion original (no deforma la imagen).
- Corrige la orientacion segun los datos EXIF de la foto.
- No agranda imagenes que ya son mas chicas.
- Guarda el resultado en una carpeta aparte, sin modificar las fotos originales.
- Genera el reporte `datos/datos_procesados/reporte_redimensionamiento.csv` con el tamaño original, el tamaño nuevo y la escala de cada imagen.

## Resultados

- Primera ejecucion sobre las fotos originales: 162 imagenes (127 de training y 35 de test), sin errores.
- Todas quedaron con su lado mayor de 1024 pixeles o menos y con su proporcion original.
- Las fotos de training ya venian en baja resolucion (1600x720, 1599x1200 o 720x1600 pixeles), salvo 3 fotos de 4096x2304. Las fotos de test vienen en alta resolucion (entre 3024 y 5712 pixeles por lado).
- El script se volvio a usar despues del data augmentation (ver avance 4).
