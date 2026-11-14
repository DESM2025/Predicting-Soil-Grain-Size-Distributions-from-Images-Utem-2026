"""Prueba del método 2D a 3D (avance 9) en fotos de entrenamiento.

Para cada foto:
1. Detecta los granos y mide su área en píxeles.
2. Pasa el área a mm2 con la escala del celular (ppm ajustado).
3. Calcula el diámetro equivalente.
4. Calcula el volumen suponiendo esferas.
5. Arma la curva de la parte gruesa (granos mayores a 2 mm) y la compara
   con la curva real en 6.3, 20 y 63 mm.

Por defecto se prueban todas las fotos de las muestras H038, H181 y H183,
que son las que tienen más material grueso.

Uso (desde la raíz del proyecto):
    python scripts/prueba_metodo_2d_3d.py
    python scripts/prueba_metodo_2d_3d.py --fotos Motorola_Edge_H181_01.jpg Samsung_A52_H183_01.jpg
"""

from argparse import ArgumentParser
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
FOTOS_DIR = PROJECT_DIR / "datos" / "Training-All_Photos_updated" / "Training-All_Photos_updated"
ETIQUETAS_CSV = PROJECT_DIR / "datos" / "Training_labels_updated.csv"
PPM_CSV = PROJECT_DIR / "datos" / "ppm_updated.csv"
SALIDA_DIR = PROJECT_DIR / "resultados" / "metodo_2d_3d"

MUESTRAS_PRUEBA = ["H038", "H181", "H183"]

# Prefijo del nombre de la foto -> nombre del celular en ppm_updated.csv.
# El orden importa: "Motorola_Edge_60_fusion" debe ir antes que "Motorola_Edge".
CELULARES = [
    ("Motorola_Edge_60_fusion", "Motorola Edge 60 Fusion"),
    ("Motorola_Edge", "Motorola Edge"),
    ("Samsung_A52", "Samsung A52"),
]

TAMANO_MINIMO_MM = 2.0
TAMICES_GRUESOS = ["6.3", "20", "63"]
AREA_MINIMA_PX = 25
SEPARACION_CENTROS_PX = 21


def celular_de_la_foto(nombre: str) -> str:
    for prefijo, celular in CELULARES:
        if nombre.startswith(prefijo):
            return celular
    raise ValueError(f"No se reconoce el celular de la foto: {nombre}")


def muestra_de_la_foto(nombre: str) -> str:
    for parte in Path(nombre).stem.split("_"):
        if len(parte) == 4 and parte[0].isalpha() and parte[1:].isdigit():
            return parte
    raise ValueError(f"No se encontró el sample_id en: {nombre}")


def ppm_ajustado(imagen: np.ndarray, celular: str) -> float:
    """Paso 2: ajusta los píxeles por mm a la resolución real de la foto."""
    tabla = pd.read_csv(PPM_CSV)
    fila = tabla.loc[tabla["phone"] == celular].iloc[0]
    lado_mayor_original = max(fila["width"], fila["height"])
    lado_mayor_foto = max(imagen.shape[:2])
    return float(fila["ppm"]) * lado_mayor_foto / lado_mayor_original


def zona_de_suelo(gris: np.ndarray) -> np.ndarray:
    """Máscara del suelo, sin el marco oscuro de la caja ni la luz."""
    promedio = cv2.blur(gris, (61, 61))
    suelo = np.uint8((promedio > 125) & (promedio < 235)) * 255
    _, etiquetas, stats, _ = cv2.connectedComponentsWithStats(suelo)
    mayor = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    zona = np.uint8(etiquetas == mayor) * 255
    # Rellenar huecos y achicar un poco para no tocar el borde de la caja.
    contornos, _ = cv2.findContours(zona, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    zona = np.zeros_like(zona)
    cv2.drawContours(zona, contornos, -1, 255, thickness=cv2.FILLED)
    return cv2.erode(zona, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31)))


def detectar_granos(imagen: np.ndarray) -> tuple[list[float], np.ndarray]:
    """Paso 1: separa los granos y devuelve el área en píxeles de cada uno."""
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    zona = zona_de_suelo(gris)

    # Suavizar la textura de cada piedra sin borrar sus bordes.
    suave = cv2.bilateralFilter(gris, 9, 40, 9)
    # Los granos son más claros que lo que los rodea (sombras entre granos).
    granos = cv2.adaptiveThreshold(
        suave, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, -6
    )
    granos = cv2.bitwise_and(granos, zona)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    granos = cv2.morphologyEx(granos, cv2.MORPH_OPEN, kernel, iterations=2)
    granos = cv2.morphologyEx(granos, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Separar granos que se tocan: cada grano tiene un "centro" donde la
    # distancia al fondo es máxima, y watershed reparte el resto entre centros.
    distancia = cv2.distanceTransform(granos, cv2.DIST_L2, 5)
    vecindad = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SEPARACION_CENTROS_PX, SEPARACION_CENTROS_PX))
    centros = np.uint8((cv2.dilate(distancia, vecindad) == distancia) & (distancia >= 3))
    centros = cv2.dilate(centros, kernel, iterations=2)
    _, marcadores = cv2.connectedComponents(centros)
    marcadores = marcadores + 1  # 1 = fondo
    marcadores[(granos > 0) & (centros == 0)] = 0  # zona a repartir entre granos
    marcadores = cv2.watershed(cv2.cvtColor(suave, cv2.COLOR_GRAY2BGR), marcadores.astype(np.int32))

    areas = []
    dibujo = imagen.copy()
    borde_zona = cv2.subtract(zona, cv2.erode(zona, kernel, iterations=3))
    for etiqueta in np.unique(marcadores):
        if etiqueta <= 1:
            continue
        mascara = np.uint8((marcadores == etiqueta) & (granos > 0))
        area = int(mascara.sum())
        if area < AREA_MINIMA_PX:
            continue
        # Granos cortados por el borde de la zona de suelo: se descartan.
        if np.any(mascara[borde_zona > 0]):
            continue
        areas.append(float(area))
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(dibujo, contornos, -1, (0, 0, 255), 1)
    contorno_zona, _ = cv2.findContours(zona, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(dibujo, contorno_zona, -1, (255, 255, 0), 2)
    return areas, dibujo


def curva_gruesa_estimada(diametros_mm: np.ndarray) -> dict[str, float]:
    """Pasos 4 y 5: volumen de esferas y % que pasa, solo granos > 2 mm."""
    gruesos = diametros_mm[diametros_mm > TAMANO_MINIMO_MM]
    volumenes = np.pi / 6 * gruesos**3
    total = volumenes.sum()
    return {
        tamiz: float(volumenes[gruesos <= float(tamiz)].sum() / total * 100) if total else np.nan
        for tamiz in TAMICES_GRUESOS
    }


def curva_gruesa_real(etiqueta: pd.Series) -> dict[str, float]:
    """Curva real de la parte gruesa: P grueso(D) = (P(D) - P(2)) / (100 - P(2)) x 100."""
    p2 = float(etiqueta["2"])
    return {
        tamiz: (float(etiqueta[tamiz]) - p2) / (100 - p2) * 100
        for tamiz in TAMICES_GRUESOS
    }


def main() -> None:
    parser = ArgumentParser(description="Prueba del método 2D a 3D en fotos de entrenamiento.")
    parser.add_argument("--fotos", nargs="+", default=None)
    argumentos = parser.parse_args()
    fotos = argumentos.fotos or sorted(
        foto.name
        for foto in FOTOS_DIR.iterdir()
        if any(f"_{muestra}_" in foto.name for muestra in MUESTRAS_PRUEBA)
    )

    SALIDA_DIR.mkdir(parents=True, exist_ok=True)
    etiquetas = pd.read_csv(ETIQUETAS_CSV).set_index("sample_id")

    filas = []
    curvas = []
    for nombre in fotos:
        imagen = cv2.imread(str(FOTOS_DIR / nombre))
        if imagen is None:
            raise FileNotFoundError(f"No se pudo leer la foto: {nombre}")
        celular = celular_de_la_foto(nombre)
        muestra = muestra_de_la_foto(nombre)
        ppm = ppm_ajustado(imagen, celular)

        areas_px, dibujo = detectar_granos(imagen)
        areas_mm2 = np.array(areas_px) / ppm**2
        diametros_mm = 2 * np.sqrt(areas_mm2 / np.pi)

        estimada = curva_gruesa_estimada(diametros_mm)
        real = curva_gruesa_real(etiquetas.loc[muestra])
        errores = {tamiz: abs(estimada[tamiz] - real[tamiz]) for tamiz in TAMICES_GRUESOS}
        mae = float(np.mean(list(errores.values())))

        cv2.imwrite(
            str(SALIDA_DIR / f"granos_detectados_{Path(nombre).stem}.jpg"),
            dibujo,
            [cv2.IMWRITE_JPEG_QUALITY, 85],
        )
        curvas.append((muestra, real, estimada))

        for tamiz in TAMICES_GRUESOS:
            filas.append(
                {
                    "foto": nombre,
                    "muestra": muestra,
                    "celular": celular,
                    "pixeles_por_mm": round(ppm, 3),
                    "granos_detectados": len(areas_px),
                    "granos_mayores_2mm": int((diametros_mm > TAMANO_MINIMO_MM).sum()),
                    "tamiz_mm": tamiz,
                    "real_pct": round(real[tamiz], 2),
                    "estimada_pct": round(estimada[tamiz], 2),
                    "error_abs": round(errores[tamiz], 2),
                }
            )

        print(
            f"{nombre}: {len(areas_px)} granos detectados, "
            f"{int((diametros_mm > TAMANO_MINIMO_MM).sum())} mayores a 2 mm, "
            f"{ppm:.2f} píxeles por mm"
        )
        for tamiz in TAMICES_GRUESOS:
            print(
                f"  {tamiz:>4} mm: real {real[tamiz]:6.2f} %  "
                f"estimada {estimada[tamiz]:6.2f} %  error {errores[tamiz]:6.2f}"
            )
        print(f"  Error promedio (MAE parte gruesa): {mae:.2f} puntos")

    resultados = pd.DataFrame(filas)
    resultados.to_csv(SALIDA_DIR / "resultados_metodo_2d_3d.csv", index=False)

    # Gráfico: una curva real por muestra y la estimada de cada una de sus fotos.
    muestras = sorted({muestra for muestra, _, _ in curvas})
    figura, ejes = plt.subplots(1, len(muestras), figsize=(5.5 * len(muestras), 4.5), squeeze=False)
    x = [float(t) for t in TAMICES_GRUESOS]
    for eje, muestra in zip(ejes[0], muestras):
        de_la_muestra = [(real, estimada) for m, real, estimada in curvas if m == muestra]
        for indice, (_, estimada) in enumerate(de_la_muestra):
            eje.semilogx(
                x, [estimada[t] for t in TAMICES_GRUESOS], "s--", color="#eb6834",
                alpha=0.5, linewidth=1, label="Método 2D a 3D (cada foto)" if indice == 0 else None,
            )
        real = de_la_muestra[0][0]
        eje.semilogx(x, [real[t] for t in TAMICES_GRUESOS], "o-", color="#2a78d6", linewidth=2.5, label="Curva real")
        mae_muestra = resultados.loc[resultados["muestra"] == muestra, "error_abs"].mean()
        eje.set_title(f"Muestra {muestra} ({len(de_la_muestra)} fotos)\nMAE parte gruesa: {mae_muestra:.2f} puntos", fontsize=10)
        eje.set_xlabel("Tamaño de tamiz (mm)")
        eje.set_ylabel("% que pasa (solo granos > 2 mm)")
        eje.set_xticks(x, TAMICES_GRUESOS)
        eje.set_ylim(0, 105)
        eje.grid(True, alpha=0.3)
        eje.legend(fontsize=8)
    figura.tight_layout()
    figura.savefig(SALIDA_DIR / "curvas_metodo_2d_3d.png", dpi=150)
    plt.close(figura)

    print("\nError promedio por muestra (MAE parte gruesa):")
    for muestra in muestras:
        print(f"  {muestra}: {resultados.loc[resultados['muestra'] == muestra, 'error_abs'].mean():.2f} puntos")
    print(f"\nError promedio de todas las fotos: {resultados['error_abs'].mean():.2f} puntos")
    print(f"Resultados guardados en: {SALIDA_DIR}")


if __name__ == "__main__":
    main()
