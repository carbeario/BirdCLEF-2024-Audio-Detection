import os
import torch
import pandas as pd
from pathlib import Path
import streamlit as st




# ==========================================
# 0. ESTILOS PÁGINAS
# ==========================================
def aplicar_estilos():
    st.markdown("""
        <style>
        /* Ocultar menú de hamburguesa y footer de Streamlit */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Botones más bonitos (con sombra y bordes redondeados) */
        .stButton>button {
            border-radius: 8px;
            font-weight: bold;
            border: none;
            box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: scale(1.02);
            box-shadow: 0px 4px 8px rgba(0,0,0,0.2);
        }
        
        /* Cajas de métricas con fondo suave */
        div[data-testid="stMetric"] {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 10px;
            border-left: 5px solid #1f77b4;
        }
        </style>
    """, unsafe_allow_html=True)




# ==========================================
# 1. GESTIÓN DE RUTAS
# ==========================================
# Resolvemos la ruta absoluta para que funcione en cualquier PC
BASE_DIR = Path(__file__).resolve().parent.parent 

ASSETS_DIR = BASE_DIR / "assets"
WEIGHTS_DIR = BASE_DIR / "weights"
PAGES_DIR = BASE_DIR / "pages"
SRC_DIR = BASE_DIR / "src"
TEMP_DIR = BASE_DIR / "temp"

# Crear carpetas necesarias
TEMP_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# Archivos específicos
LOGO_PATH = ASSETS_DIR / "logo.png"
# IMPORTANTE: Asegúrate de que el archivo .pth se llame así en tu carpeta /weights
MODEL_PATH = WEIGHTS_DIR / "panns_inference.pth" 
METADATA_PATH = ASSETS_DIR / "species_metadata.csv"


# ==========================================
# 2. LA VERDAD ABSOLUTA (LISTA MAESTRA)
# ==========================================
# Esta lista define el orden EXACTO de las neuronas de salida del modelo.
# NO MODIFICAR EL ORDEN BAJO NINGÚN CONCEPTO.
SPECIES_LIST = [
    'asbfly', 'ashdro1', 'ashpri1', 'asikoe2', 'barswa', 'bcnher', 'bkskit1', 'bkwsti', 
    'bladro1', 'blakit1', 'blhori1', 'blnmon1', 'blrwar1', 'brcful1', 'brnhao1', 'brnshr', 
    'brodro1', 'brwowl1', 'btbeat1', 'categr', 'cohcuc1', 'comgre', 'comior1', 'comkin1', 
    'commoo3', 'commyn', 'compea', 'comros', 'comsan', 'comtai1', 'copbar1', 'crseag1', 
    'eaywag1', 'emedov2', 'eucdov', 'eurcoo', 'gargan', 'gloibi', 'graher1', 'grbeat1', 
    'grecou1', 'greegr', 'grefla1', 'grejun2', 'grewar3', 'grnsan', 'grnwar1', 'grtdro1', 
    'gryfra', 'grywag', 'gybpri1', 'gyhcaf1', 'hoopoe', 'houcro1', 'houspa', 'inbrob1', 
    'indpit1', 'insbab1', 'junbab2', 'kenplo1', 'labcro1', 'laudov1', 'lblwar1', 'lirplo', 
    'litegr', 'litgre1', 'litspi1', 'litswi1', 'marsan', 'nutman', 'oripip1', 'piebus1', 
    'piekin1', 'plapri1', 'purher1', 'pursun3', 'pursun4', 'putbab1', 'rerswa1', 'revbul', 
    'rewbul', 'rewlap1', 'rocpig', 'rorpar', 'rossta2', 'ruftre2', 'shikra1', 'spodov', 
    'stbkin1', 'thbwar1', 'tibfly3', 'wemhar1', 'whbbul2', 'whbwat1', 'whbwoo2', 'whcbar1', 
    'whiter2', 'whrmun', 'whtkin2', 'woosan', 'zitcis1'
]

# Mapa inverso para búsquedas rápidas (Etiqueta -> Índice)
SPECIES_TO_ID = {label: i for i, label in enumerate(SPECIES_LIST)}


# ==========================================
# 3. CARGA DE METADATOS (SINCRONIZADA)
# ==========================================
def load_species_data():
    """
    Carga el CSV y lo reordena para que coincida EXACTAMENTE con SPECIES_LIST.
    Esto evita errores si el CSV se ordena alfabéticamente por nombre común, etc.
    """
    if METADATA_PATH.exists():
        try:
            df = pd.read_csv(METADATA_PATH)
            df['primary_label'] = df['primary_label'].astype(str)
            
            # --- BLINDAJE DE SEGURIDAD ---
            # 1. Filtramos solo las especies que el modelo conoce
            df = df[df['primary_label'].isin(SPECIES_LIST)]
            
            # 2. Establecemos el 'primary_label' como índice y reordenamos según la lista maestra
            df = df.set_index('primary_label')
            df = df.reindex(SPECIES_LIST)
            df = df.reset_index() # Recuperamos la columna
            
            return df
        except Exception as e:
            print(f"❌ Error leyendo CSV: {e}")
            return pd.DataFrame() # Devuelve vacío si falla
    else:
        print("⚠️ ALERTA: No se encuentra species_metadata.csv.")
        return pd.DataFrame()

# DataFrame disponible para toda la app
SPECIES_DF = load_species_data()


# ==========================================
# 4. CONFIGURACIÓN DEL MODELO (PANNs)
# ==========================================
# Parámetros de Audio
SAMPLE_RATE = 32000
DURATION = 5            # Segundos
FMIN = 20
FMAX = 16000 # Frecuencia máxima para PANNs (importante)
HOP_LENGTH = 512

# Configuración Específica de PANNs
MODEL_CONFIG = {
    "n_mels": 64,       # PANNs usa 64
    "name": "PANNs (CNN14)",
    "description": "Red Convolucional optimizada con SpecAugment."
}

# Hardware: Detecta automáticamente si tienes GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# 5. UI / ESTÉTICA
# ==========================================
PROJECT_TITLE = "BirdCLEF: Reconocimiento de Aves"
PROJECT_ICON = "🦅"
COLOR_PRIMARY = "#1f77b4"