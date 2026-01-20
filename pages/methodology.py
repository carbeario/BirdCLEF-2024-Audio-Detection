import streamlit as st

# 1. CONFIGURACIÓN DE PÁGINA
st.set_page_config(
    page_title="Metodología Técnica",
    page_icon="🧠",
    layout="wide"
)

# 2. CSS ESPECÍFICO (SOLO LO NECESARIO)
st.markdown("""
    <style>
        /* A. ARREGLAR LAS MÉTRICAS (Imprescindible para el contraste) */
        div[data-testid="stMetric"] {
            background-color: #262730; /* Fondo oscuro tarjeta */
            border: 1px solid #464b5c;
            padding: 15px !important;
            border-radius: 10px;
        }
        div[data-testid="stMetricValue"] {
            color: #FFFFFF !important; /* Texto blanco brillante */
            font-weight: 700;
        }
        div[data-testid="stMetricLabel"] {
            color: #b0b3c5 !important;
        }
        div[data-testid="stMetricDelta"] {
            background-color: #1e3a2f;
            padding: 2px 8px;
            border-radius: 5px;
            margin-bottom: 5px;
        }

        /* B. HE ELIMINADO EL CSS DE LAS PESTAÑAS (TABS) PARA QUE SE VEAN NORMALES */

        /* C. CONTENEDORES DE TEXTO (Estilo consistente) */
        .info-box {
            background-color: #151925;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #FF9F33; /* Acento naranja */
            margin-bottom: 20px;
        }
        
        /* D. Tarjeta de Hardware (Reemplaza al st.code feo) */
        .hardware-card {
            background-color: #0e1117;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 15px;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
            color: #00FF00; /* Estilo terminal hacker suave */
        }
    </style>
""", unsafe_allow_html=True)

# 3. CONTENIDO
st.title("Arquitectura y Metodología")
st.markdown("<div style='font-size: 1.1rem; color: #a3a8b8; margin-bottom: 20px;'>Detalles técnicos del pipeline de entrenamiento e inferencia implementado en este TFG.</div>", unsafe_allow_html=True)

# --- SECCIÓN 1: PIPELINE ---
st.header("1. Pipeline de Procesamiento")
st.markdown("El sistema sigue un flujo de datos estricto para transformar el audio crudo en probabilidades:")

# Diagrama Graphviz (Mantenemos la mejora de contraste)
st.graphviz_chart("""
    digraph {
        rankdir=LR;
        bgcolor="transparent"; 
        node [shape=box, style="filled,rounded", fontname="Sans-Serif", fontcolor="black", penwidth=0];
        edge [color="#a3a8b8", arrowsize=0.8];

        Audio [label="Audio Raw\n(32kHz)", fillcolor="#e0e0e0"];
        Preproc [label="Preprocesamiento\n(5s Crop/Pad)", fillcolor="#FFD700"];
        Mel [label="Log-Mel\nSpectrogram\n(64 bins)", fillcolor="#FFD700"];
        Augment [label="SpecAugment\n(Solo Train)", fillcolor="#FF6B6B", fontcolor="white"];
        CNN [label="PANNs\n(CNN14)", fillcolor="#4DA6FF", fontcolor="white"];
        FC [label="Capa Densa\n(101 Neuronas)", fillcolor="#4DA6FF", fontcolor="white"];
        Output [label="Softmax\nProbabilidades", fillcolor="#90EE90"];
        
        Audio -> Preproc -> Mel -> Augment -> CNN -> FC -> Output;
    }
""")

st.markdown("---")

# --- SECCIÓN 2: ARQUITECTURA ---
st.header("2. Arquitectura del Modelo")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("El Modelo: PANNs (CNN14)")
    st.markdown("""
    <div class="info-box">
    Se ha seleccionado la arquitectura <b>CNN14</b> de la librería <i>Pretrained Audio Neural Networks</i>.
    <br><br>
    <ul>
        <li><b>Backbone:</b> Red Convolucional Profunda (14 capas).</li>
        <li><b>Pre-entrenamiento:</b> AudioSet (Google).</li>
        <li><b>Adaptación:</b> Head customizado para <b>101 especies</b>.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.subheader("Preprocesamiento")
    st.markdown("""
    <div class="info-box" style="border-left-color: #4DA6FF;">
    Las redes neuronales "ven" el sonido. Convertimos el audio en imágenes:
    <br><br>
    <ul>
        <li><b>Sample Rate:</b> 32.000 Hz.</li>
        <li><b>Ventana:</b> 5 segundos fijos.</li>
        <li><b>Transform:</b> Log-Mel Spectrogram (64 bins).</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

st.success("💡 **Decisión de Diseño:** Se priorizó una resolución de 64 Mels frente a 128 Mels para reducir la carga computacional en un 50%.")

st.markdown("---")

# --- SECCIÓN 3: ENTRENAMIENTO (AQUÍ ESTABA EL ERROR) ---
st.header("3. Estrategia de Entrenamiento")

# Las pestañas ahora usarán el estilo por defecto de Streamlit (mucho más limpio)
tab1, tab2, tab3 = st.tabs(["🛡️ SpecAugment", "📈 OneCycleLR", "⚡ Hardware"])

with tab1:
    st.markdown("#### Robustez mediante 'Ruido Visual'")
    st.write("Se utilizó **SpecAugment**, una técnica que 'tacha' aleatoriamente franjas de tiempo y frecuencia.")
    st.info("Objetivo: Evitar que el modelo memorice silbidos específicos y entienda el contexto global.")

with tab2:
    st.markdown("#### Planificador One Cycle Policy")
    st.write("El Learning Rate empieza lento, sube rápido (exploración) y baja suave (refinamiento).")
    st.info("Resultado: Convergencia en solo **3 épocas**.")

with tab3:
    st.markdown("#### Ingeniería de Rendimiento")
    st.write("Entrenamiento local en Laptop NVIDIA RTX 3050 Ti optimizando el flujo de datos:")
    
    # OPCIÓN B: ESTILO ELEGANTE (Igual que el resto de la página)
    # Usamos la clase "info-box" que ya definimos, pero forzamos el borde rojo/rosa
    st.markdown("""
    <div class="info-box" style="border-left-color: #FF4B4B;">
    <ul>
        <li><b>Optimizer:</b> AdamW</li>
        <li><b>Precision:</b> Mixed (float16) <i style="color:#888">- Ahorro VRAM</i></li>
        <li><b>Workers:</b> 4 Hilos CPU <i style="color:#888">- Carga paralela</i></li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
st.markdown("---")

# --- SECCIÓN 4: RESULTADOS ---
st.header("4. Resultados Experimentales")

# Las métricas se ven bien gracias al CSS (Sección A)
m1, m2, m3 = st.columns(3)
m1.metric("Precisión Global (Top-1)", "61.08%", "+3.4% vs Base")
m2.metric("Precisión Top-3", "85%", "Estimada")
m3.metric("Tiempo Inferencia", "95ms", "GPU")

st.markdown("""
<div style='background-color: #1a1c24; padding: 15px; border-radius: 10px; border: 1px solid #333; margin-top: 20px; font-style: italic; color: #ccc;'>
"A pesar de las limitaciones de hardware y la complejidad del dataset (101 clases desbalanceadas), el modelo PANNs superó a arquitecturas más complejas como PaSST (Transformers) en este escenario."
</div>
""", unsafe_allow_html=True)