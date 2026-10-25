### Integración del aumento de datos y preprocesamiento

Se implementó un script de data augmentation para ampliar las 127 imágenes originales del conjunto training. Por cada imagen se conservaron la versión original y cinco variantes generadas mediante volteos horizontales y verticales, rotaciones de 90°, 180° y 270°, variaciones leves de brillo y contraste, y ruido gaussiano aplicado aleatoriamente (probabilidad de ruido = 0.30 ; desviacion = 4.0).

Las etiquetas granulométricas asociadas a cada sample_id se conservaron en todas las variantes. Como resultado, el conjunto de entrenamiento aumentó a 762 imágenes y se generó el archivo dataset_final_aumentado.csv.

Posteriormente, se integró el script de redimensionamiento anterior para reducir las imágenes a un máximo de 1024 píxeles en su dimensión mayor, manteniendo la relación de aspecto y corrigiendo la orientación cuando fue necesario. Las imágenes aumentadas de entrenamiento y las imágenes originales de prueba se procesaron por separado.

El procesamiento final generó 762 imágenes redimensionadas en datos_procesados/training/ y 35 imágenes de prueba en datos_procesados/test/, sin modificar los archivos originales. También se creó un reporte con las dimensiones y escalas aplicadas a cada imagen.

El flujo de preprocesamiento quedó establecido de la siguiente manera:

```text
Imágenes originales de training
        ↓
Data augmentation
        ↓
762 imágenes aumentadas
        ↓
Redimensionamiento
        ↓
datos_procesados/training/

Imágenes originales de test
        ↓
Redimensionamiento
        ↓
datos_procesados/test/
```
