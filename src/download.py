import sys
import requests
import shutil
import time
from pathlib import Path

# Configuración de rutas
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

try:
    from src import config
except ImportError:
    print("❌ ERROR: No se pudo importar config.py.")
    sys.exit(1)

BASE_IMAGES_DIR = config.ASSETS_DIR / "images"
PATH_DEFAULT_LOGO = config.ASSETS_DIR / "logo.png"

def get_inaturalist_image(scientific_name):
    """Obtiene la 'foto por defecto' curada por la comunidad en alta calidad."""
    url = f"https://api.inaturalist.org/v1/taxa?q={scientific_name}"
    headers = {"User-Agent": "BirdCLEF_2024_TFG_App/1.0"}
    
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            results = res.json().get("results", [])
            if results:
                # Extraemos la foto por defecto del primer resultado coincidente
                photo_data = results[0].get("default_photo")
                if photo_data:
                    return photo_data.get("medium_url")
    except Exception as e:
        print(f"   ❌ Error en API iNaturalist para {scientific_name}")
    return None

def main():
    print("\n--- INICIANDO DESCARGA DESDE iNATURALIST (Alta Calidad) ---")
    BASE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    df = config.SPECIES_DF
    if df.empty:
        print("❌ ERROR: No hay especies cargadas.")
        return

    total = len(df)
    descargadas = 0
    
    for i, (_, row) in enumerate(df.iterrows()):
        bird_id = row['primary_label']
        scientific = row['scientific_name']
        
        bird_dir = BASE_IMAGES_DIR / bird_id
        bird_dir.mkdir(parents=True, exist_ok=True)
        photo_path = bird_dir / "photo.jpg"
        
        if photo_path.exists():
            continue
            
        print(f"[{i+1}/{total}] Descargando '{scientific}'...")
        img_url = get_inaturalist_image(scientific)
        
        exito = False
        if img_url:
            try:
                # Descargamos el archivo físico
                img_response = requests.get(img_url, stream=True, timeout=10)
                if img_response.status_code == 200:
                    with open(photo_path, 'wb') as f:
                        shutil.copyfileobj(img_response.raw, f)
                    descargadas += 1
                    exito = True
            except requests.exceptions.RequestException:
                pass
                
        # Fallback local seguro
        if not exito and PATH_DEFAULT_LOGO.exists():
            shutil.copy(PATH_DEFAULT_LOGO, photo_path)
            
        # Freno de 1.5s para no superar las 60 peticiones/minuto de iNaturalist
        time.sleep(1.5)

    print(f"--- LISTO: {descargadas} fotos de alta calidad guardadas ---\n")

if __name__ == "__main__":
    main()