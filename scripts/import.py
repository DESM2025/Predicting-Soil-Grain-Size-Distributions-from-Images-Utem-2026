import os
import zipfile


project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_local_env() -> None:
    env_path = os.path.join(project_dir, ".env")
    if not os.path.isfile(env_path):
        return

    with open(env_path, encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            name = name.strip()
            value = value.strip().strip("\"'")
            if name:
                os.environ.setdefault(name, value)


load_local_env()

from kaggle.api.kaggle_api_extended import KaggleApi

if not os.environ.get("KAGGLE_API_TOKEN"):
    raise RuntimeError(
        "Falta la variable de entorno KAGGLE_API_TOKEN. "
        "Configúrala antes de ejecutar este script."
    )

# Autenticar API
api = KaggleApi()
api.authenticate()

competition_name = "soil-grain-size-from-photos"  
descarga_dir = os.path.join(project_dir, "datos")

# Crear la carpeta "datos" si no existe
os.makedirs(descarga_dir, exist_ok=True)

# Descargar datos de la competencia
print(f"Carpeta lista en: {descarga_dir}")
print(f"Iniciando descarga de {competition_name}...")
api.competition_download_files(competition_name, path=descarga_dir, force=False, quiet=False)

# Descomprimir el archivo .zip 
zip_path = os.path.join(descarga_dir, f"{competition_name}.zip")

if os.path.exists(zip_path):
    print("Descomprimiendo")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(descarga_dir)
    
    os.remove(zip_path)
    print("Descargado")
else:
    print("Descarga finalizada.")