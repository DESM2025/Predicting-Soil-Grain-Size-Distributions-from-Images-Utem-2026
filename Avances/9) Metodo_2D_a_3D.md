# 9) Metodo para estimar tamaño y volumen de particulas a partir de su area 2D

En la foto vemos los granos desde arriba, un area (2D). Pero la curva granulometrica se mide en peso, y el peso depende del volumen (3D). Este metodo explica como pasar de lo que se ve en la foto a la curva.

### Paso 1: medir cada grano en la foto

Con OpenCV se separan los granos del fondo y cuenta cuantos pixeles ocupa cada uno. Ese numero es el area del grano en pixeles.

Se descartan los granos cortados por el borde de la foto y las manchas muy chicas, porque darian medidas falsas.

### Paso 2: pasar de pixeles a milimetros

El dataset trae la tabla ppm_updated.csv que dice cuantos pixeles hay en 1 mm para cada celular. Como las fotos de training estan achicadas hay que ajustar ese valor:

    pixeles por mm = valor de la tabla x (ancho de la foto / ancho original del celular)

Por que: si una foto se achica a la mitad de su ancho, 1 mm del suelo ocupa la mitad de pixeles. Por eso el valor de la tabla se multiplica por cuanto se achico la foto.

Ejemplo con el Motorola Edge: la tabla dice 11.492 pixeles por mm para fotos de 4000 pixeles de ancho. Las fotos de training miden 1600 de ancho, entonces 11.492 x 1600 / 4000 = 4.6 pixeles por mm. Con el Samsung A52 da casi lo mismo: 26.33 x 1599 / 9248 = 4.55.

Con eso:

    area en mm2 = area en pixeles / (pixeles por mm)^2

Por que se eleva al cuadrado: un area tiene ancho y alto. Si 1 mm son 4.6 pixeles de ancho y 4.6 de alto, entonces 1 mm2 son 4.6 x 4.6 = 21.2 pixeles.

### Paso 3: sacar el tamaño del grano

Suponemos que el grano es redondo y calculamos el diametro de un circulo con esa misma area:

    diametro = 2 x raiz(area / 3.1416)

De donde sale: el area de un circulo es area = pi x radio^2. Despejando el radio:

    radio = raiz(area / pi)

Y como el diametro es el doble del radio:

    diametro = 2 x raiz(area / pi)

con pi = 3.1416.

### Paso 4: calcular el volumen (de 2D a 3D)

Como no vemos el grosor del grano, suponemos que es una esfera:

    volumen = 0.5236 x diametro^3

De donde sale: el volumen de una esfera es volumen = 4/3 x pi x radio^3. Como medimos el diametro y el radio es la mitad (radio = diametro / 2):

    volumen = 4/3 x pi x (diametro / 2)^3
            = 4/3 x pi x diametro^3 / 8
            = pi / 6 x diametro^3

Y pi / 6 = 3.1416 / 6 = 0.5236.

### Paso 5: armar la curva

Para cada tamiz (2, 6.3, 20, 63 y 200 mm) se calcula que porcentaje del volumen total corresponde a granos mas chicos que ese tamiz:

    % que pasa = volumen de los granos mas chicos que el tamiz / volumen de todos los granos x 100

Por que se puede usar el volumen en vez del peso: el peso de un grano es su densidad por su volumen (peso = densidad x volumen). Si suponemos que todos los granos tienen la misma densidad, al calcular el porcentaje la densidad aparece arriba y abajo de la division y se cancela. Por eso el porcentaje en volumen es igual al porcentaje en peso.

Se usa el volumen y no la cantidad de granos porque un grano grande pesa mucho mas que muchos granos chicos: como el volumen depende del diametro al cubo, un grano de 10 mm tiene el mismo volumen que 1000 granos de 1 mm.

## Ejemplo con un grano

- Ocupa 2000 pixeles en la foto.
- Area: 2000 / 4.6^2 = 94.5 mm2.
- Diametro: 2 x raiz(94.5 / 3.1416) = 10.97 mm.
- Volumen: 0.5236 x 10.97^3 = 691 mm3.

Como mide entre 6.3 y 20 mm, cuenta para los tamices de 20 mm hacia arriba.

## Supuestos

- La foto se tomo desde arriba y sin inclinacion.
- Los granos son redondos (esferas).
- Todos los granos pesan lo mismo por unidad de volumen.
- Los granos que se ven en la foto representan a toda la muestra.

## Limitaciones

- En las fotos de training cada pixel mide 0.22 mm, asi que los granos menores a 1 mm casi no se ven. El metodo solo sirve para los de 2 mm hacia arriba.
- Solo se ven los granos de arriba. Los que estan tapados no se cuentan.
- Si dos granos estan pegados, el programa puede contarlos como uno solo.
- Las sombras pueden confundirse con granos.
- Hay pocos granos grandes por foto, asi que la medicion de los tamaños grandes puede variar bastante.

## Como probarlo/siguiente paso

1. Usar las fotos originales de datos/Training-All_Photos_updated/.
2. Elegir muestras con bastante material grueso: H181, H183 o H038.
3. Como el metodo no ve los granos finos, comparar solo la parte gruesa de la curva (6.3, 20 y 63 mm). Para que ambas curvas sean comparables, en las dos se cuentan solo los granos mayores a 2 mm.
4. Reportar la diferencia promedio entre la curva del metodo y la curva real en esos tamaños.

## Para que sirve

El modelo se equivoca mas en los tamaños grandes (entre 14 y 17 puntos desde 6.3 mm). Si este metodo mide mejor esos tamaños, se pueden usar sus resultados en vez de los del modelo para esa parte de la curva.
