import streamlit as st
import torch
import numpy as np
import pandas as pd
import time
import requests
import os

# Importamos tus módulos locales
from src import config, inference, audio_utils

# ==========================================
# 1. CONFIGURACIÓN DE PÁGINA
# ==========================================
st.set_page_config(page_title="Detector en Vivo", page_icon="🎙️", layout="wide")

# ==========================================
# 2. CSS "HIGH-END" (ESTILOS PODIO)
# ==========================================
st.markdown("""
    <style>
        /* A. CONTENEDOR PRINCIPAL */
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }
        #MainMenu, footer {visibility: hidden;}

        /* B. TARJETAS DE RESULTADOS */
        .result-card {
            background-color: #1a1c24;
            border: 1px solid #3c4043;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            margin-bottom: 20px;
            border-left: 5px solid #FF4B4B; /* Acento rojo para el ganador */
        }
        
        /* C. TEXTOS */
        h1 {
            background: linear-gradient(90deg, #ffffff, #888888);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .species-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 5px;
        }
        .species-sci {
            font-size: 1.1rem;
            font-style: italic;
            color: #a0a0a0;
            margin-bottom: 15px;
        }

        /* D. BARRAS DE PROGRESO (PODIO) */
        .bar-container {
            width: 100%;
            background-color: #2c2f38;
            height: 12px; /* Altura más fina y elegante */
            border-radius: 6px;
            overflow: hidden;
            margin-top: 5px;
        }
        .bar-fill {
            height: 100%;
            border-radius: 6px;
            transition: width 0.6s ease-in-out;
        }
        
        /* E. COLORES DEL PODIO */
        .rank-1 { background: linear-gradient(90deg, #FF4B4B, #FF914D); } /* Rojo-Naranja (Ganador) */
        .rank-2 { background: linear-gradient(90deg, #FFD700, #FDB931); } /* Dorado (2º) */
        .rank-3 { background: linear-gradient(90deg, #00BFFF, #1E90FF); } /* Azul (3º) */
        
        /* F. FILAS SECUNDARIAS */
        .secondary-row {
            margin-bottom: 12px;
            padding: 8px 12px;
            background-color: #262730;
            border-radius: 8px;
            border: 1px solid #333;
        }

    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. FUNCIONES AUXILIARES (UI)
# ==========================================
def get_wiki_image_url(scientific_name):
    """Busca foto en Wikipedia (versión ligera)"""
    try:
        query = scientific_name.replace(" ", "_")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query}"
        resp = requests.get(url, headers={'User-Agent': 'BirdCLEF-Demo/1.0'}, timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            if 'originalimage' in data: return data['originalimage']['source']
            if 'thumbnail' in data: return data['thumbnail']['source']
    except:
        pass
    return "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/No_image_available.svg/600px-No_image_available.svg.png"

# ==========================================
# 4. INTERFAZ PRINCIPAL
# ==========================================

# --- CABECERA ---
c1, c2 = st.columns([3, 1], vertical_alignment="center")
with c1:
    st.title("🎙️ Detector Bioacústico")
    st.markdown("Sube un audio y selecciona la Inteligencia Artificial que prefieras.")

# --- SELECTOR DE MODELO (NUEVO) ---
with c2:
    model_option = st.radio(
        "🧠 Cerebro AI:",
        ("PANNs (Rápido)", "PaSST (Preciso)")
    )
    # Traducir a clave interna
    model_key = 'panns' if 'PANNs' in model_option else 'passt'
    
    # Semáforo simple
    path_existe = os.path.exists(config.MODEL_PATHS[model_key])
    if path_existe:
        st.success("Listo", icon="✅")
    else:
        st.warning("Descargando...", icon="⏳")

st.markdown("---")

# ==========================================
# ZONA DE ENTRADA (PESTAÑAS)
# ==========================================
final_audio_path = None 

tab_upload, tab_sample = st.tabs(["📂 Subir Archivo Propio", "🎵 Probar Ejemplos"])

# --- OPCIÓN A: UPLOAD ---
with tab_upload:
    uploaded_file = st.file_uploader("", type=["wav", "mp3", "ogg"], label_visibility="collapsed")
    if uploaded_file:
        temp_path = os.path.join(config.TEMP_DIR, uploaded_file.name)
        os.makedirs(config.TEMP_DIR, exist_ok=True) # Asegurar que existe temp
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        final_audio_path = temp_path

# --- OPCIÓN B: EJEMPLOS ---
with tab_sample:
    SAMPLES = {
        "Selecciona un ejemplo...": None,
        "Asian Brown Flycatcher": "asbfly.mp3",
        "Red-whiskered Bulbul": "rewbul.wav", 
        "White-throated Kingfisher": "whtkin.mp3"
    }
    
    selected_sample = st.selectbox("Elige un audio de la galería:", list(SAMPLES.keys()))
    
    if SAMPLES[selected_sample] is not None:
        file_name = SAMPLES[selected_sample]
        # Ajusta la ruta si tus samples están en otra carpeta
        full_path = os.path.join(config.ASSETS_DIR, "samples", file_name)
        
        # Fallback por si la estructura de carpetas varía
        if not os.path.exists(full_path):
             full_path = os.path.join("assets", "samples", file_name)

        if os.path.exists(full_path):
            final_audio_path = full_path
        else:
            st.error(f"⚠️ Archivo no encontrado: {full_path}")

# ==========================================
# VISUALIZACIÓN Y ANÁLISIS
# ==========================================
if final_audio_path:
    with st.container(border=True):
        col_audio, col_spec = st.columns([1, 2], gap="large")
        
        with col_audio:
            st.subheader("1. Escuchar")
            st.caption(f"Archivo: {os.path.basename(final_audio_path)}")
            st.audio(final_audio_path)
            
            st.markdown("<br>", unsafe_allow_html=True)
            # BOTÓN DE ANÁLISIS
            analyze_btn = st.button(f"ANALIZAR CON {model_option.upper()}", type="primary", use_container_width=True)

        with col_spec:
            st.subheader("2. Visión del Modelo")
            try:
                fig_spec = audio_utils.generar_espectrograma_visual(final_audio_path)
                if fig_spec:
                    st.pyplot(fig_spec, use_container_width=True)
                else:
                    st.info("No se pudo generar la vista previa.")
            except Exception as e:
                st.warning("Vista previa no disponible.")

    # ==========================================
    # LÓGICA DE PREDICCIÓN
    # ==========================================
    if analyze_btn:
        st.markdown("---")
        
        progress_text = "Procesando..."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(0, 80, 20):
            time.sleep(0.05)
            my_bar.progress(percent_complete + 10, text=f"Cargando {model_option}...")

        # --- INFERENCIA ---
        # Ahora devuelve un array con 101 probabilidades
        all_probs = inference.predict_bird(final_audio_path, model_key)
        
        my_bar.progress(100, text="¡Finalizado!")
        time.sleep(0.2)
        my_bar.empty()

        if all_probs is not None:
            # Convertimos a tensor para usar torch.topk (es más cómodo)
            probs_tensor = torch.tensor(all_probs)
            
            # Obtenemos Top 3
            top3_prob, top3_idx = torch.topk(probs_tensor, 3)
            
            top3_prob = top3_prob.numpy()
            top3_idx = top3_idx.numpy()
            
            # --- DATOS DEL GANADOR ---
            winner_idx = top3_idx[0]
            winner_prob = top3_prob[0] * 100
            winner_label = config.SPECIES_LIST[winner_idx]
            
            # Recuperar info del CSV
            try:
                row = config.SPECIES_DF[config.SPECIES_DF['primary_label'] == winner_label].iloc[0]
                common_name = row['common_name']
                sci_name = row['scientific_name']
            except:
                common_name = winner_label
                sci_name = "Desconocido"

            # --- MOSTRAR RESULTADOS ---
            r_col1, r_col2 = st.columns([1, 2], gap="large")
            
            with r_col1:
                with st.container(border=True):
                    img_url = get_wiki_image_url(sci_name)
                    st.image(img_url, use_container_width=True)
                    st.caption("Fuente: Wikipedia API")

            with r_col2:
                # --- PUESTO 1 (ORO) ---
                html_card = f"""
                <div class="result-card">
                    <div style="color: #FF9F33; font-weight: bold; letter-spacing: 1px; margin-bottom: 5px;">🏆 RESULTADO PRINCIPAL</div>
                    <div class="species-title">{common_name}</div>
                    <div class="species-sci">{sci_name}</div>
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 5px;">
                        <span style="font-size: 1.5rem; font-weight: 800; color: #FF4B4B;">{winner_prob:.1f}%</span>
                        <span style="color: #888; font-size: 0.8rem;">Confianza</span>
                    </div>
                    <div class="bar-container">
                        <div class="bar-fill rank-1" style="width: {winner_prob}%;"></div>
                    </div>
                </div>
                """
                st.markdown(html_card, unsafe_allow_html=True)
                
                st.markdown("#### Otras Posibilidades")
                
                # --- PUESTO 2 y 3 (PLATA y BRONCE) ---
                for i in range(1, 3):
                    idx = top3_idx[i]
                    prob = top3_prob[i] * 100
                    lbl = config.SPECIES_LIST[idx]
                    
                    try:
                        name = config.SPECIES_DF[config.SPECIES_DF['primary_label'] == lbl].iloc[0]['common_name']
                    except:
                        name = lbl
                    
                    # Generamos la clase CSS correcta (rank-2 o rank-3)
                    rank_class = f"rank-{i+1}"
                    
                    st.markdown(f"""
                    <div class="secondary-row">
                        <div style="display:flex; justify-content:space-between; font-weight:500; margin-bottom:4px; color:#EEE;">
                            <span>{i+1}. {name}</span>
                            <span>{prob:.1f}%</span>
                        </div>
                        <div class="bar-container">
                            <div class="bar-fill {rank_class}" style="width: {prob}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        else:
            st.error("Error al procesar el modelo. Revisa los logs en la terminal.")