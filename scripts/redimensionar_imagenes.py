
from pathlib import Path
from PIL import Image, ImageOps
import pandas as pd

# CONFIGURACIÓN

BASE = Path(__file__).resolve().parent.parent

CARPETA_DESTINO = BASE / "datos" / "datos_procesados"

DATASETS = [
    (
        "training",
        BASE / "datos" / "fotos_aumentadas",
        CARPETA_DESTINO / "training",
    ),
    (
        "test",
        BASE / "datos" / "Test_All_Photos",
        CARPETA_DESTINO / "test",
    ),
]

MAX_RESOLUCION = 1024

EXTENSIONES = {".jpg", ".jpeg", ".png", ".webp"}

# REDIMENSIONAR UNA IMAGEN

def redimensionar_imagen(origen, destino, carpeta_origen):

    with Image.open(origen) as imagen:

        # Respetar orientación original de la foto
        imagen = ImageOps.exif_transpose(imagen)

        ancho_original, alto_original = imagen.size

        # Mantener proporciones y evitar ampliar imágenes
        imagen.thumbnail(
            (MAX_RESOLUCION, MAX_RESOLUCION),
            Image.Resampling.LANCZOS
        )

        ancho_nuevo, alto_nuevo = imagen.size

        # Crear carpeta de destino
        destino.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # Guardar imagen redimensionada
        if destino.suffix.lower() in [".jpg", ".jpeg"]:
            imagen = imagen.convert("RGB")
            imagen.save(destino, quality=95)
        else:
            imagen.save(destino)

        return {
            "archivo": str(origen.relative_to(carpeta_origen)),
            "ancho_original": ancho_original,
            "alto_original": alto_original,
            "ancho_nuevo": ancho_nuevo,
            "alto_nuevo": alto_nuevo,
            "escala_x": ancho_nuevo / ancho_original,
            "escala_y": alto_nuevo / alto_original
        }


# PROCESAR DATASET

def procesar_dataset():

    registros = []
    errores = []

    for nombre_dataset, carpeta_origen, carpeta_destino in DATASETS:
        if not carpeta_origen.exists():
            raise FileNotFoundError(
                f"No se encontró la carpeta de {nombre_dataset}: {carpeta_origen}"
            )

        imagenes = sorted(
            archivo
            for archivo in carpeta_origen.rglob("*")
            if archivo.is_file()
            and archivo.suffix.lower() in EXTENSIONES
        )

        print(f"\nCarpeta de origen ({nombre_dataset}): {carpeta_origen}")
        print(f"Imágenes encontradas: {len(imagenes)}")

        for i, archivo in enumerate(imagenes, start=1):
            ruta_relativa = archivo.relative_to(carpeta_origen)
            salida = carpeta_destino / ruta_relativa

            try:
                info = redimensionar_imagen(
                    archivo,
                    salida,
                    carpeta_origen,
                )
                info["conjunto"] = nombre_dataset
                registros.append(info)

            except Exception as error:
                errores.append({
                    "conjunto": nombre_dataset,
                    "archivo": str(archivo),
                    "error": str(error)
                })

            if i % 50 == 0 or i == len(imagenes):
                print(f"Procesadas: {i}/{len(imagenes)}")

    # GUARDAR RESULTADOS

    CARPETA_DESTINO.mkdir(
        parents=True,
        exist_ok=True
    )

    reporte = pd.DataFrame(registros)

    reporte.to_csv(
        CARPETA_DESTINO / "reporte_redimensionamiento.csv",
        index=False
    )

    if errores:
        pd.DataFrame(errores).to_csv(
            CARPETA_DESTINO / "errores.csv",
            index=False
        )

    print("\nPROCESAMIENTO FINALIZADO")
    print("Imágenes procesadas:", len(registros))
    print("Errores:", len(errores))
    print("Destino:", CARPETA_DESTINO)


if __name__ == "__main__":
    procesar_dataset()