# 10) Prueba y conclusion del metodo 2D a 3D

Se llevo a cabo el metodo del avance 9 en prueba_metodo_2d_3d.py y se probo en 21 fotos de las 3 muestras con mas material grueso. Para cada foto se midieron los granos, se calculo su volumen suponiendo esferas y se comparo la parte gruesa de la curva (6.3, 20 y 63 mm) con la curva real. Los resultados estan en resultados/metodo_2d_3d/.

| Muestra | Fotos | Error promedio (MAE) |
|---|---|---|
| H181 | 6 | 4.50 |
| H183 | 7 | 5.05 |
| H038 | 8 | 37.15 |

En H181 y H183 el metodo funciona bien. En H038 falla porque mas del 35 % de su material son piedras de mas de 20 mm que no aparecen en las fotos, y el metodo no puede medir lo que no se ve. Ademas, en ninguna foto se detectaron granos de mas de 20 mm, y los granos menores a 2 mm no se ven por la resolucion de las fotos.

Conclusion: el metodo solo sirve para gravas de 2 a 20 mm cuando la foto representa bien la muestra. Como el modelo EfficientNet ya predice la curva completa, se recomienda no usarlo en la submission y dejarlo como experimento documentado y posible mejora futura con fotos de mayor resolucion.
