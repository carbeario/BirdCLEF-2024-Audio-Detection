import streamlit as st
from PIL import Image
import os
from src import config

# 1. CONFIGURACIÓN
st.set_page_config(layout="wide", page_title="BirdCLEF 2024 - TFG", page_icon="🦅")

# 2. CSS DE ALTA PRECISIÓN
st.markdown("""
    <style>
        /* A. CONTENEDOR PRINCIPAL */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 3rem !important;
        }
        #MainMenu, footer {visibility: hidden;}

        /* B. TEXTOS Y TÍTULOS */
        h1 {
            font-size: 3rem !important;
            font-weight: 800;
            background: linear-gradient(90deg, #4da9ff, #ff9f33); /* Azul a Naranja */
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0 !important;
        }
        .subtitle {
            font-size: 1.2rem;
            color: #b0b3b8;
            margin-bottom: 20px;
        }
        
        /* C. CAJAS DE MÉTRICAS */
        div[data-testid="stMetric"] {
            background-color: #1a1c24;
            border: 1px solid #2d2f36;
            border-radius: 12px;
            padding: 15px !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            color: #ffffff !important;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.9rem !important;
            color: #888 !important;
        }

        /* D. TARJETAS DE ARQUITECTURA */
        .arch-card {
            background-color: #1e1e1e;
            border-radius: 15px;
            padding: 25px;
            height: 100%;
            border: 1px solid #333;
            transition: transform 0.2s;
        }
        .arch-card:hover {
            border-color: #555;
            transform: translateY(-5px);
        }
        .arch-title {
            font-size: 1.3rem;
            font-weight: bold;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .arch-desc {
            font-size: 0.95rem;
            color: #cccccc;
            line-height: 1.5;
        }
        .tag {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: bold;
            margin-top: 10px;
            margin-right: 5px;
        }
        .tag-red { background-color: rgba(255, 75, 75, 0.2); color: #ff4b4b; }
        .tag-blue { background-color: rgba(0, 191, 255, 0.2); color: #00bfff; }

        /* E. SEPARADOR */
        .separator {
            height: 1px;
            background: linear-gradient(90deg, transparent, #444, transparent);
            margin: 40px 0;
        }
        
        /* F. IMAGEN */
        img {
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }
    </style>
""", unsafe_allow_html=True)

# --- HERO SECTION ---
c_img, c_text = st.columns([1, 2.5], gap="large", vertical_alignment="center")

with c_img:
    if os.path.exists("assets/logo.png"):
        st.image("assets/logo.png", use_container_width=True)
    else:
        st.info("Logo no encontrado")

with c_text:
    st.markdown("<h1>BirdCLEF 2024</h1>", unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Sistema de Monitorización de Biodiversidad mediante <b>Deep Learning Dual</b></p>', unsafe_allow_html=True)
    
    st.markdown("""
    Este proyecto de Fin de Grado (TFG) implementa una solución SOTA (*State-of-the-Art*) para la clasificación de **101 especies de aves**.
    
    El sistema integra dos paradigmas de Inteligencia Artificial para equilibrar precisión y velocidad:
    1.  **PANNs (CNN14)** para inferencia rápida en dispositivos ligeros.
    2.  **PaSST (Patchout Audio Spectrogram Transformer)** para máxima precisión y reducción de complejidad computacional.
    """)

# --- KPIs ---
st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Especies Clasificadas", "101", "Clases Balanceadas")
kpi2.metric("Dataset Procesado", "17.942", "Espectrogramas Mel")
kpi3.metric("Precisión PaSST", "74.7%", "Transformer (SOTA)")
kpi4.metric("Precisión PANNs", "60.1%", "CNN (Ligero)")

# --- ARQUITECTURA DUAL ---
st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
st.subheader("Arquitectura del Sistema")

col_panns, col_passt = st.columns(2, gap="medium")

with col_panns:
    st.markdown("""
    <div class="arch-card">
        <div class="arch-title" style="color: #FF4B4B;">
            PANNs (Modelo Rápido)
        </div>
        <div class="arch-desc">
            Basado en la arquitectura <b>CNN14</b> (ResNet-like). 
            Está diseñado para dispositivos con recursos limitados. 
            Procesa espectrogramas de 64 Mels.
        </div>
        <div style="margin-top:15px;">
            <span class="tag tag-red">Baja Latencia</span>
            <span class="tag tag-red">CPU Friendly</span>
            <span class="tag tag-red">Robustez Básica</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_passt:
    st.markdown("""
    <div class="arch-card">
        <div class="arch-title" style="color: #00BFFF;">
            PaSST (Modelo Preciso)
        </div>
        <div class="arch-desc">
            Implementa <b>Patchout Audio Spectrogram Transformer</b>. 
            Utiliza "Patchout" para reducir la complejidad del entrenamiento y mecanismos de atención para captar patrones globales.
        </div>
        <div style="margin-top:15px;">
            <span class="tag tag-blue">Alta Precisión (+14%)</span>
            <span class="tag tag-blue">Efficient Transformer</span>
            <span class="tag tag-blue">Atención Global</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- CTA ---
st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
st.info("**Navega al menú lateral** para probar el **Detector en Vivo** o explorar el **Catálogo de Especies**.")