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
        
        /* Botones más bonitos */
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
        
        /* Cajas de métricas */
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
BASE_DIR = Path(__file__).resolve().parent.parent 

ASSETS_DIR = BASE_DIR / "assets"
WEIGHTS_DIR = BASE_DIR / "weights"
PAGES_DIR = BASE_DIR / "pages"
SRC_DIR = BASE_DIR / "src"
TEMP_DIR = BASE_DIR / "temp"

# Crear carpetas necesarias si no existen
TEMP_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# Archivos específicos
LOGO_PATH = ASSETS_DIR / "logo.png"
METADATA_PATH = ASSETS_DIR / "species_metadata.csv"
PANNS_LABELS_PATH = ASSETS_DIR / "class_labels_indices.csv" # <--- NUEVO

# --- DICCIONARIO DE MODELOS ---
MODEL_PATHS = {
    'panns': WEIGHTS_DIR / "panns_mejor.pth",       # Tu fine-tuning
    'passt': WEIGHTS_DIR / "passt_mejor.pth",       # Tu fine-tuning
    'panns_base': WEIGHTS_DIR / "Cnn14_mAP=0.431.pth" # <--- NUEVO: La base
}


# ==========================================
# 2. LISTA MAESTRA
# ==========================================
# Lista de 101 especies (Orden de entrenamiento)
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

# Mapa inverso (Etiqueta -> Índice)
SPECIES_TO_ID = {label: i for i, label in enumerate(SPECIES_LIST)}

# ==========================================
# 3. CARGA DE METADATOS
# ==========================================
def load_species_data():
    if METADATA_PATH.exists():
        try:
            df = pd.read_csv(METADATA_PATH)
            df['primary_label'] = df['primary_label'].astype(str)
            df = df[df['primary_label'].isin(SPECIES_LIST)]
            df = df.set_index('primary_label')
            df = df.reindex(SPECIES_LIST)
            df = df.reset_index()
            return df
        except Exception as e:
            print(f"Error leyendo CSV: {e}")
            return pd.DataFrame()
    else:
        # print("ALERTA: No se encuentra species_metadata.csv.")
        return pd.DataFrame()

SPECIES_DF = load_species_data()

# ==========================================
# 4. CONFIGURACIÓN DE AUDIO
# ==========================================
SAMPLE_RATE = 32000
DURATION = 5
FMIN = 20
FMAX = 14000  # Ajustado a 14000 para coincidir con el entrenamiento
HOP_LENGTH = 512

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 5. UI / ESTÉTICA
# ==========================================
PROJECT_TITLE = "BirdCLEF 2024: Detector Avanzado"
PROJECT_ICON = "🦅"
COLOR_PRIMARY = "#1f77b4"