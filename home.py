import streamlit as st

# 1. CONFIGURACIÓN
st.set_page_config(layout="wide", page_title="BirdCLEF 2024", page_icon="🦜")

# 2. CSS DE ALTA PRECISIÓN
st.markdown("""
    <style>
        /* A. CONTENEDOR PRINCIPAL */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 4rem !important;
            padding-left: 3rem;
            padding-right: 3rem;
        }

        /* B. OCULTAR ELEMENTOS SOBRANTES */
        #MainMenu, footer {visibility: hidden;}
        
        /* C. CONTROL DEL ESPACIO VERTICAL */
        div[data-testid="stVerticalBlock"] > div {
            gap: 1.5rem !important;
        }

        /* D. ESTILO DE LA IMAGEN */
        img {
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
            object-fit: cover;
        }

        /* E. CAJAS DE MÉTRICAS (COLUMNA 1) */
        div[data-testid="stMetric"] {
            background-color: #202124;
            border: 1px solid #3c4043;
            border-radius: 10px;
            padding: 10px !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            height: 140px !important; /* Altura fija para alinear */
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        div[data-testid="stMetricValue"] {
            font-size: 2rem !important;
            color: #FFFFFF !important;
            font-weight: 700;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.9rem !important;
            color: #9aa0a6 !important;
        }
        div[data-testid="stMetricDelta"] {
            margin-bottom: 5px;
        }

        /* F. TÍTULOS */
        h1 {
            font-size: 2.8rem !important;
            margin-bottom: 5px !important;
            background: linear-gradient(90deg, #ffffff, #888888);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        h3 {
            font-size: 1rem !important;
            color: #FF9F33;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 15px !important;
            margin-top: 10px !important;
        }

        /* G. DESCRIPCIÓN BOX (HERO) */
        .desc-box {
            background-color: #131d2e;
            border-left: 4px solid #3b82f6;
            padding: 20px;
            border-radius: 0 8px 8px 0;
            margin-top: 15px;
            line-height: 1.6;
        }
        
        /* H. TARJETAS DE INFORMACIÓN (NUEVO PARA COL 2 Y 3) */
        .info-card {
            background-color: #202124; /* Mismo color que métricas */
            border: 1px solid #3c4043;
            border-radius: 10px;
            padding: 20px;
            height: 140px; /* MISMA ALTURA QUE MÉTRICAS */
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            display: flex;
            flex-direction: column;
            justify-content: center; /* Centra el texto verticalmente */
        }

        /* I. LISTAS DENTRO DE LAS TARJETAS */
        .tech-list {
            font-size: 0.95rem;
            color: #bdc1c6;
            line-height: 1.6; /* Interlineado normal y legible */
            margin: 0;
        }
        .tech-list b { color: #e8eaed; }
        
        /* J. LINEA SEPARADORA */
        .separator {
            border: 0;
            height: 1px;
            background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(255, 255, 255, 0.3), rgba(0, 0, 0, 0));
            margin-top: 40px;
            margin-bottom: 40px;
        }
    </style>
""", unsafe_allow_html=True)

# --- FILA SUPERIOR (HERO) ---
c_img, c_text = st.columns([1.2, 3], gap="large", vertical_alignment="center")

with c_img:
    st.image("assets/logo.png", use_container_width=True)

with c_text:
    st.markdown("<h1>BirdCLEF 2024</h1>", unsafe_allow_html=True)
    st.markdown("<div style='color: #9aa0a6; font-size: 1.1rem; margin-bottom: 5px;'>Sistema de Reconocimiento Bioacústico Avanzado (TFG)</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="desc-box">
    <span style="color: #3b82f6; font-weight: bold;">Descripción:</span> 
    Plataforma basada en <b style="color:white">Deep Learning (CNN14)</b> diseñada para identificar biodiversidad en tiempo real. 
    Clasifica cantos de aves entre <b style="color:white">101 especies</b> analizando espectrogramas visuales.
    </div>
    """, unsafe_allow_html=True)

# --- LÍNEA SEPARADORA ---
st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

# --- FILA INFERIOR (ESTRUCTURA DE TARJETAS) ---
col1, col2, col3 = st.columns([1, 1.2, 1.2], gap="medium")

with col1:
    st.subheader("🚀 RENDIMIENTO")
    # Métricas (Ya tienen estilo de caja por CSS)
    m1, m2 = st.columns(2, gap="small")
    m1.metric("Top-1", "61%", "+ Test Set") 
    m2.metric("Top-3", "85%")
    
    st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)
    st.caption("Hardware: NVIDIA RTX | Framework: PyTorch")

with col2:
    st.subheader("🛠 STACK TECNOLÓGICO")
    # AQUI ESTÁ EL CAMBIO: Envolvemos en un div .info-card
    st.markdown("""
    <div class="info-card">
        <div class="tech-list">
            <b>Modelo:</b> PANNs (CNN14 Pre-trained)<br>
            <b>Optimización:</b> SpecAugment & OneCycleLR<br>
            <b>Input:</b> Espectrogramas Mel (Log-mel)<br>
            <b>Frontend:</b> Streamlit + CSS Custom
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.subheader("⚙️ FLUJO DE PROCESO")
    # AQUI ESTÁ EL CAMBIO: Envolvemos en un div .info-card
    st.markdown("""
    <div class="info-card">
        <div class="tech-list">
        1️⃣ <b>Captura:</b> Audio Raw (WAV/MP3)<br>
        2️⃣ <b>Pre-proceso:</b> STFT a Imagen<br>
        3️⃣ <b>Inferencia:</b> Extracción rasgos (CNN)<br>
        4️⃣ <b>Salida:</b> Vector probabilidades
        </div>
    </div>
    """, unsafe_allow_html=True)