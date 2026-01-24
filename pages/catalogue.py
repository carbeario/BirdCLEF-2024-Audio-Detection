import streamlit as st
import pandas as pd
import requests
import re
import html
from src import config

# 1. CONFIGURACIÓN
st.set_page_config(page_title="Catálogo de Especies", page_icon="📖", layout="wide")

# 2. CSS ESTILO "LIMPIO"
st.markdown("""
    <style>
        /* A. MÉTRICAS CON ESTILO */
        div[data-testid="stMetric"] {
            background-color: #1a1c24;
            border: 1px solid #333;
            padding: 15px !important;
            border-radius: 10px;
            text-align: center; /* Centrado para que quede elegante */
        }
        div[data-testid="stMetricValue"] {
            color: #4da9ff !important; /* Azul técnico */
            font-size: 1.4rem !important;
            font-weight: 700;
        }
        div[data-testid="stMetricLabel"] {
            color: #a0a0a0 !important;
            font-size: 0.9rem !important;
        }

        /* B. CAJA DE DESCRIPCIÓN */
        .desc-box {
            background-color: #13151b;
            border-left: 4px solid #4da9ff; /* Azul a juego con la métrica */
            padding: 20px;
            border-radius: 0 8px 8px 0;
            color: #e0e0e0;
            line-height: 1.6;
            margin-top: 15px;
            font-size: 0.95rem;
        }

        /* C. ID TAG */
        .id-tag {
            background-color: #333;
            color: #ff9f33; /* Naranja para resaltar el código */
            padding: 3px 8px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 0.9em;
            border: 1px solid #555;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. LÓGICA (SOLO LO ÚTIL)
# ==========================================
@st.cache_data(ttl=3600*24, show_spinner=False)
def get_bird_details(scientific_name):
    # Valores por defecto
    result = {
        "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/No_image_available.svg/600px-No_image_available.svg.png",
        "extract": "Descripción no disponible en la API de Wikipedia.",
        "familia": "Taxonomía no disponible"
    }
    
    if not scientific_name: return result

    query = scientific_name.replace(" ", "_")
    headers = {'User-Agent': 'BirdCLEF-Student-App/1.0'}

    # 1. API REST (Para Imagen y Resumen)
    try:
        api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query}"
        resp = requests.get(api_url, headers=headers, timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            # Imagen
            if 'originalimage' in data: result["img"] = data['originalimage']['source']
            elif 'thumbnail' in data: result["img"] = data['thumbnail']['source']
            # Texto
            if 'extract' in data: result["extract"] = data['extract']
    except:
        pass

    # 2. SCRAPING HTML (SOLO PARA FAMILIA - RELEVANTE CIENTÍFICAMENTE)
    try:
        html_url = f"https://en.wikipedia.org/wiki/{query}"
        html_resp = requests.get(html_url, headers=headers, timeout=4)
        if html_resp.status_code == 200:
            full_html = html_resp.text
            clean_text = html.unescape(full_html) 
            
            # Buscar Familia (Ej: Muscicapidae)
            match = re.search(r'Family:.*?([A-Z][a-z]+idae)', clean_text, re.IGNORECASE | re.DOTALL)
            if match:
                result["familia"] = match.group(1)
            else:
                match_simple = re.search(r'\b([A-Z][a-z]+idae)\b', result["extract"])
                if match_simple: result["familia"] = match_simple.group(1)

    except Exception as e:
        print(f"Error scraping wiki: {e}")

    return result

# ==========================================
# 4. INTERFAZ
# ==========================================
if config.SPECIES_DF.empty:
    st.error("⚠️ No se pudo cargar el archivo species_metadata.csv")
    st.stop()

# --- SIDEBAR (BUSCADOR) ---
with st.sidebar:
    st.header("🔍 Buscador")
    st.info(f"Total especies: **{len(config.SPECIES_DF)}**")
    
    opciones = [f"{r['common_name']} ({r['primary_label']})" for _, r in config.SPECIES_DF.iterrows()]
    seleccion = st.selectbox("Selecciona una ave:", opciones)
    codigo = seleccion.split("(")[-1].replace(")", "")

# --- LÓGICA PRINCIPAL ---
row = config.SPECIES_DF[config.SPECIES_DF['primary_label'] == codigo].iloc[0]
sci_name = row['scientific_name']
com_name = row['common_name']
link_xc = row['url_xenocanto']

# Carga de datos
with st.spinner("Cargando ficha..."):
    wiki_data = get_bird_details(sci_name)

# --- LAYOUT LIMPIO ---
c_head, c_btn = st.columns([3, 1], vertical_alignment="center")
with c_head:
    st.markdown(f"# {com_name}")
    st.markdown(f"**Científico:** *{sci_name}* &nbsp; | &nbsp; **ID:** <span class='id-tag'>{codigo}</span>", unsafe_allow_html=True)

with c_btn:
    st.link_button("🔊 Escuchar Audio", link_xc, use_container_width=True)

st.markdown("---")

col_img, col_info = st.columns([1, 1.5], gap="large")

with col_img:
    st.image(wiki_data["img"], use_container_width=True)

with col_info:
    # SECCIÓN TÉCNICA (SOLO FAMILIA)
    # Al quitar el status, la familia destaca más como dato biológico clave
    st.metric("Familia Taxonómica", wiki_data['familia'])
    
    st.markdown("#### Descripción")
    if wiki_data['extract']:
        st.markdown(f'<div class="desc-box">{wiki_data["extract"]}</div>', unsafe_allow_html=True)
    
    st.caption(f"ℹ️ Fuente: Wikipedia API.")