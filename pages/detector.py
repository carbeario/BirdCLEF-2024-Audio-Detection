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
# 2. CSS "HIGH-END" (ESTILOS)
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

        /* D. BARRAS DE PROGRESO PERSONALIZADAS */
        .progress-container {
            width: 100%;
            background-color: #2c2f38;
            border-radius: 8px;
            margin: 10px 0;
            height: 25px; /* Altura estándar */
            position: relative;
            overflow: hidden; /* Para que el borde redondeado corte la barra */
        }
        .progress-bar {
            height: 100%;
            border-radius: 8px;
            text-align: right;
            line-height: 25px; /* Centrar texto verticalmente */
            color: white;
            padding-right: 10px;
            font-size: 0.85rem;
            font-weight: bold;
            transition: width 1s ease-in-out;
            min-width: 2%; /* Para que siempre se vea un poquito */
        }
        /* Colores para Top 1, 2 y 3 */
        .bar-1 { background: linear-gradient(90deg, #FF4B4B, #FF914D); }
        .bar-2 { background-color: #3b82f6; }
        .bar-3 { background-color: #64748b; }

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
    st.markdown("Sube un audio o selecciona un ejemplo y el modelo **CNN14** identificará la especie.")
with c2:
    with st.spinner("Conectando neuronas..."):
        model = inference.cargar_modelo()
    
    if model:
        st.success(f"Modelo Activo ({config.DEVICE.upper()})", icon="🧠")
    else:
        st.error("Error de Modelo", icon="🚨")
        st.stop()

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
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        final_audio_path = temp_path

# --- OPCIÓN B: EJEMPLOS ---
with tab_sample:
    # ⚠️ ASEGÚRATE DE QUE ESTOS NOMBRES EXISTAN EN 'assets/samples/'
    SAMPLES = {
        "Selecciona un ejemplo...": None,
        "Asian Brown Flycatcher": "asbfly.mp3",
        "Red-whiskered Bulbul": "rewbul.wav", 
        "White-throated Kingfisher": "whtkin.mp3"
    }
    
    selected_sample = st.selectbox("Elige un audio de la galería:", list(SAMPLES.keys()))
    
    if SAMPLES[selected_sample] is not None:
        file_name = SAMPLES[selected_sample]
        full_path = os.path.join("assets", "samples", file_name)
        
        if os.path.exists(full_path):
            final_audio_path = full_path
        else:
            st.error(f"⚠️ Archivo no encontrado: {full_path}. Revisa la carpeta assets/samples.")

# ==========================================
# VISUALIZACIÓN Y ANÁLISIS
# ==========================================
if final_audio_path:
    # Contenedor unificado
    with st.container(border=True):
        col_audio, col_spec = st.columns([1, 2], gap="large")
        
        with col_audio:
            st.subheader("1. Escuchar")
            st.caption(f"Archivo: {os.path.basename(final_audio_path)}")
            st.audio(final_audio_path)
            
            st.markdown("<br>", unsafe_allow_html=True)
            # BOTÓN DE ANÁLISIS
            analyze_btn = st.button("🦅 ANALIZAR ESPECIE", type="primary", use_container_width=True)

        with col_spec:
            st.subheader("2. Visión del Modelo")
            try:
                # Generar espectrograma visual
                fig_spec = audio_utils.generar_espectrograma_visual(final_audio_path)
                st.pyplot(fig_spec, use_container_width=True)
            except Exception as e:
                st.warning("Vista previa del espectrograma no disponible.")

    # ==========================================
    # LÓGICA DE PREDICCIÓN
    # ==========================================
    if analyze_btn:
        st.markdown("---")
        
        # Barra de progreso "Fake" para UX
        progress_text = "Procesando..."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(0, 100, 20):
            time.sleep(0.05)
            my_bar.progress(percent_complete + 10, text="Extrayendo patrones Mel...")

        # --- INFERENCIA REAL ---
        probs = inference.predecir(model, final_audio_path)
        
        my_bar.progress(100, text="¡Finalizado!")
        time.sleep(0.2)
        my_bar.empty()

        if probs is not None:
            # Procesar Resultados
            if not isinstance(probs, torch.Tensor):
                probs = torch.tensor(probs)
                
            top3_prob, top3_idx = torch.topk(probs, 3)
            
            top3_prob = top3_prob.cpu().numpy()
            top3_idx = top3_idx.cpu().numpy()
            
            # Datos del Ganador
            winner_idx = top3_idx[0]
            winner_label = config.SPECIES_LIST[winner_idx]
            winner_prob_val = top3_prob[0]*100
            
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
                # IMAGEN WIKI
                with st.container(border=True):
                    img_url = get_wiki_image_url(sci_name)
                    st.image(img_url, use_container_width=True)
                    st.caption("Fuente: Wikipedia API")

            with r_col2:
                # TARJETA PRINCIPAL (SIN SANGRÍA PARA EVITAR ERROR HTML)
                html_card = f"""
<div class="result-card">
<div style="color: #FF9F33; font-weight: bold; letter-spacing: 1px; margin-bottom: 5px;">RESULTADO PRINCIPAL</div>
<div class="species-title">{common_name}</div>
<div class="species-sci">{sci_name}</div>
<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
<span style="font-size: 2.5rem; font-weight: 800; color: #FF4B4B;">{winner_prob_val:.1f}%</span>
<span style="color: #888;">Certeza del modelo</span>
</div>
<div class="progress-container" style="height: 12px; background-color: #333; margin-top: 0;">
<div class="progress-bar bar-1" style="width: {winner_prob_val}%;"></div>
</div>
</div>
"""
                st.markdown(html_card, unsafe_allow_html=True)
                
                st.subheader("Otras Posibilidades")
                
                # BARRAS DE PROGRESO (TOP 2 y 3)
                for i in range(1, 3):
                    idx = top3_idx[i]
                    prob = top3_prob[i] * 100
                    lbl = config.SPECIES_LIST[idx]
                    try: 
                        name = config.SPECIES_DF[config.SPECIES_DF['primary_label'] == lbl].iloc[0]['common_name']
                    except: 
                        name = lbl
                    
                    st.markdown(f"""
<div style="margin-bottom: 10px;">
<div style="display:flex; justify-content:space-between; font-size:0.9rem; color:#e0e0e0;">
<span>{i+1}. {name}</span>
<span>{prob:.1f}%</span>
</div>
<div class="progress-container">
<div class="progress-bar bar-{i+1}" style="width: {prob}%;"></div>
</div>
</div>
""", unsafe_allow_html=True)

        else:
            st.error("No se pudo procesar el audio. Verifica el formato.")