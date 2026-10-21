import os
import re
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATOS_DIR = os.path.join(PROJECT_DIR, "datos")

CARPETA_FOTOS_ORIGINALES = os.path.join(
    DATOS_DIR,
    "Training-All_Photos_updated",
    "Training-All_Photos_updated",
)
CSV_GRANULOMETRIA = os.path.join(DATOS_DIR, "Training_labels_updated.csv")
CARPETA_SALIDA = os.path.join(DATOS_DIR, "fotos_aumentadas")
CSV_SALIDA = os.path.join(DATOS_DIR, "dataset_final_aumentado.csv")

# Con 5 variantes: 127 fotos x (1 original + 5 aumentadas) = 762 fotos totales
VARIANTES_POR_FOTO = 5

# Configuración del ruido 
PROBABILIDAD_RUIDO = 0.30   # 30% de probabilidad de ruido
DESVIACION_RUIDO = 4.0      # Ruido leve (escala 0-255, simula ISO de sensor)

os.makedirs(CARPETA_SALIDA, exist_ok=True)

df_curvas = pd.read_csv(CSV_GRANULOMETRIA)
# Asegurar que la columna del ID se llame 'sample_id' (o tomar la primera columna)
col_id = df_curvas.columns[0]
df_curvas[col_id] = df_curvas[col_id].astype(str).str.strip()

# Convertir el DataFrame a un diccionario para búsqueda por ID
curvas_dict = df_curvas.set_index(col_id).to_dict(orient="index")

archivos_en_carpeta = [f for f in os.listdir(CARPETA_FOTOS_ORIGINALES) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
total_fotos = len(archivos_en_carpeta)

print(f"Se encontraron {total_fotos} fotos en '{CARPETA_FOTOS_ORIGINALES}'.")
print(f"Iniciando\n")

registros_finales = []
procesadas = 0
omitidas = 0

for nombre_foto in archivos_en_carpeta:
    # Extraer el sample_id del nombre del archivo ('H374', 'F827', 'G190')
    # Busca una letra mayúscula seguida de 3 dígitos
    coincidencia = re.search(r'([A-Z]\d{3})', nombre_foto)
    
    if not coincidencia:
        print(f"[ALERTA] No se encontró código de muestra en: {nombre_foto} (Omitida)")
        omitidas += 1
        continue
        
    sample_id = coincidencia.group(1)
    
    # Buscar la curva granulométrica de esta muestra
    if sample_id not in curvas_dict:
        print(f"[ALERTA] El código {sample_id} de '{nombre_foto}' no existe en el CSV (Omitida)")
        omitidas += 1
        continue
        
    datos_curva = curvas_dict[sample_id]
    
    # Cargar la foto original
    ruta_origen = os.path.join(CARPETA_FOTOS_ORIGINALES, nombre_foto)
    img = Image.open(ruta_origen).convert("RGB")
    nombre_base, extension = os.path.splitext(nombre_foto)
    
    # Guardar la foto original 
    nombre_original_salida = f"{nombre_base}_original{extension}"
    img.save(os.path.join(CARPETA_SALIDA, nombre_original_salida), quality=95)
    
    fila_orig = {"archivo": nombre_original_salida, "sample_id": sample_id}
    fila_orig.update(datos_curva)
    registros_finales.append(fila_orig)
    
    # Generar las variantes
    for i in range(1, VARIANTES_POR_FOTO + 1):
        img_var = img.copy()
        
        # Flips horizontal y vertical
        if random.random() > 0.5:
            img_var = img_var.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:
            img_var = img_var.transpose(Image.FLIP_TOP_BOTTOM)
            
        # Rotaciones en ángulos ortogonales (no dejan bordes negros)
        rot = random.choice([None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270])
        if rot is not None:
            img_var = img_var.transpose(rot)
            
        # Variaciones leves de iluminación (±10% de brillo y contraste)
        factor_brillo = random.uniform(0.90, 1.10)
        factor_contraste = random.uniform(0.90, 1.10)
        img_var = ImageEnhance.Brightness(img_var).enhance(factor_brillo)
        img_var = ImageEnhance.Contrast(img_var).enhance(factor_contraste)
        
        # Inyección leve de ruido gaussiano 
        arr = np.array(img_var, dtype=np.float32)
        if random.random() < PROBABILIDAD_RUIDO:
            ruido = np.random.normal(0.0, DESVIACION_RUIDO, arr.shape)
            arr = np.clip(arr + ruido, 0.0, 255.0)
            
        img_var = Image.fromarray(arr.astype(np.uint8))
        
        # Guardar variante
        nombre_var = f"{nombre_base}_aug_{i}{extension}"
        img_var.save(os.path.join(CARPETA_SALIDA, nombre_var), quality=95)
        
        # Guardar fila con la misma etiqueta granulométrica
        fila_var = {"archivo": nombre_var, "sample_id": sample_id}
        fila_var.update(datos_curva)
        registros_finales.append(fila_var)
        
    procesadas += 1
    if procesadas % 10 == 0 or procesadas == total_fotos:
        print(f"Progreso: {procesadas}/{total_fotos} fotos procesadas...")

# Guardar el nuevo CSV consolidado
df_salida = pd.DataFrame(registros_finales)
df_salida.to_csv(CSV_SALIDA, index=False)

print(f"Fotos originales procesadas con éxito: {procesadas}")
if omitidas > 0:
    print(f"Fotos omitidas por discrepancias: {omitidas}")
print(f"Total de fotos finales generadas: {len(df_salida)} en '{CARPETA_SALIDA}/'")
print(f"Nuevo CSV guardado: '{CSV_SALIDA}'")