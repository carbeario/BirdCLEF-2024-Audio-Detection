import streamlit as st

# 1. CONFIGURACIÓN DE PÁGINA
st.set_page_config(
    page_title="Metodología Técnica",
    page_icon="🧠",
    layout="wide"
)

# 2. CSS ESPECÍFICO
st.markdown("""
    <style>
        /* A. MÉTRICAS HIGH-CONTRAST */
        div[data-testid="stMetric"] {
            background-color: #262730;
            border: 1px solid #464b5c;
            padding: 15px !important;
            border-radius: 10px;
        }
        div[data-testid="stMetricValue"] {
            color: #4da9ff !important; /* Azul eléctrico */
            font-weight: 700;
        }
        div[data-testid="stMetricLabel"] {
            color: #b0b3c5 !important;
        }

        /* B. CAJAS DE INFORMACIÓN */
        .info-box {
            background-color: #151925;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #FF9F33;
            margin-bottom: 20px;
            height: 100%; /* Para igualar alturas */
        }
        
        /* C. TERMINAL STYLE */
        .terminal-box {
            font-family: 'Courier New', monospace;
            background-color: #000;
            color: #0f0;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #333;
            font-size: 0.85rem;
        }
    </style>
""", unsafe_allow_html=True)

# 3. CONTENIDO
st.title("Arquitectura y Metodología")
st.markdown("<div style='font-size: 1.1rem; color: #a3a8b8; margin-bottom: 20px;'>Estudio comparativo de arquitecturas Deep Learning para bioacústica en entornos de recursos limitados.</div>", unsafe_allow_html=True)

# --- SECCIÓN 1: PIPELINE DUAL ---
st.header("1. Pipeline de Procesamiento Dual")
st.markdown("El sistema implementa una arquitectura híbrida que permite seleccionar entre velocidad (CNN) o precisión (Transformer):")


# USAMOS COLUMNAS PARA CENTRAR
col_izq, col_centro, col_der = st.columns([0.2, 3, 0.2])

with col_centro:
    st.graphviz_chart("""
        digraph {
            rankdir=LR;
            bgcolor="transparent"; 
            node [shape=box, style="filled,rounded", fontname="Sans-Serif", fontcolor="black", penwidth=0];
            edge [color="#a3a8b8", arrowsize=0.8];

            # 1. ENTRADA
            Audio [label="Audio Raw\n(Mono, 32kHz)", fillcolor="#e0e0e0"];
            
            # 2. CORTE TEMPORAL
            Preproc [label="Crop/Pad\n(5s Fijo)", fillcolor="#FFD700"];
            
            subgraph cluster_0 {
                label="Branch A: Ligero";
                style=dashed; color="#666"; fontcolor="white";
                
                # 3A. MEL + NORM
                Mel64 [label="Log-Mel (64)\n+ Norm.", fillcolor="#FFD700"];
                CNN [label="PANNs\n(CNN14)", fillcolor="#FF6B6B", fontcolor="white"];
                
                Mel64 -> CNN;
            }

            subgraph cluster_1 {
                label="Branch B: SOTA";
                style=dashed; color="#666"; fontcolor="white";
                
                # 3B. MEL + NORM
                Mel128 [label="Log-Mel (128)\n+ Norm.", fillcolor="#FFD700"];
                PaSST [label="PaSST\n(Transformer)", fillcolor="#4DA6FF", fontcolor="white"];
                
                Mel128 -> PaSST;
            }
            
            # 4. SALIDA
            Output [label="Softmax\n(Probabilidades)", fillcolor="#90EE90"];
            
            Audio -> Preproc;
            Preproc -> Mel64;
            Preproc -> Mel128;
            CNN -> Output;
            PaSST -> Output;
        }
    """, use_container_width=True)

# AÑADIMOS UNA PEQUEÑA LEYENDA TÉCNICA DEBAJO
st.caption("Detalles STFT: Window=1024, Hop=320, fmin=50Hz, fmax=14000Hz. Normalización: Instance-wise.")



st.markdown("---")

# --- SECCIÓN 2: COMPARATIVA DE MODELOS ---
st.header("2. Arquitecturas Evaluadas")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("PANNs (CNN14)")
    st.markdown("""
    <div class="info-box" style="border-left-color: #FF6B6B;">
    <b>Pretrained Audio Neural Networks</b>
    <br><br>
    Arquitectura basada en ResNet diseñada para eficiencia.
    <ul style="margin-top:10px;">
        <li><b>Entrada:</b> 64 Mel-bins.</li>
        <li><b>Ventaja:</b> Inferencia ultrarrápida (CPU Friendly).</li>
        <li><b>Limitación:</b> Menor capacidad para captar contexto global.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.subheader("PaSST (Transformer)")
    st.markdown("""
    <div class="info-box" style="border-left-color: #4DA6FF;">
    <b>Patchout Audio Spectrogram Transformer</b>
    <br><br>
    Modelo basado en mecanismos de atención (Self-Attention).
    <ul style="margin-top:10px;">
        <li><b>Entrada:</b> 128 Mel-bins (Mayor resolución).</li>
        <li><b>Ventaja:</b> +14% Precisión. Entiende secuencias complejas.</li>
        <li><b>Técnica:</b> "Patchout" para reducir coste computacional.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- SECCIÓN 3: INGENIERÍA DE ENTRENAMIENTO ---
st.header("3. Ingeniería de Entrenamiento")
st.markdown("Optimización del proceso de aprendizaje para maximizar el rendimiento en pocas épocas (Low-Budget Training).")

tab1, tab2, tab3 = st.tabs(["PaSST (VRAM Solution)", "SpecAugment", "OneCycleLR (Scheduler)"])

# PESTAÑA 1: GRADIENT ACCUMULATION (La que ya tenías bien)
with tab1:
    st.markdown("#### Gradient Accumulation (Solo PaSST)")
    st.warning("Desafío: El Transformer (PaSST) agotaba los 4GB de VRAM con un Batch Size > 4.")
    st.write("Solución: Se implementó **Acumulación de Gradientes** para desacoplar el tamaño del batch lógico del físico.")
    
    st.markdown("""
    <div class="terminal-box">
    # Configuración de Memoria
    REAL_BATCH_SIZE = 2     # Límite físico de la GPU (RTX 3050 Ti)
    ACCUM_STEPS = 16        # Acumulaciones antes del optimizer.step()
    -----------------------
    VIRTUAL_BATCH = 32      # Batch efectivo para la estabilidad matemática
    </div>
    """, unsafe_allow_html=True)



# PESTAÑA 2: SPECAUGMENT (FIX VISUAL FINAL)
with tab2:
    st.markdown("#### Robustez mediante 'Ruido Visual'")
    
    col_desc, col_vis = st.columns([1.2, 1], gap="medium")
    
    with col_desc:
        st.write("Aplicado en el *dataloading* para evitar el overfitting.")
        st.info("💡 **Concepto Clave:** SpecAugment actúa como un 'censor'. Tapa partes aleatorias del audio para obligar a la IA a reconocer el pájaro por el contexto restante, no por un solo sonido.")
        st.markdown("""
        * **Time Masking:** Tapa un evento temporal (ej: un silbido corto).
        * **Freq Masking:** Tapa una frecuencia (ej: los graves).
        """)

    with col_vis:
        st.markdown("###### Representación Visual:")
        
        # CSS CORREGIDO
        st.markdown("""
        <style>
            .signal-bg {
                background-color: #333;
                border: 1px solid #555;
                border-radius: 6px;
                height: 60px; /* MÁS ALTURA para que quepa todo */
                position: relative;
                margin-bottom: 20px;
                display: flex;
                align-items: center;
                overflow: hidden;
            }
            .mask-label {
                color: #aaa;
                font-size: 0.75rem;
                margin-left: 15px;
                font-family: monospace;
                z-index: 10; /* IMPORTANTE: Texto por encima de la máscara */
                position: relative; /* Para que el z-index funcione */
                text-shadow: 0 1px 2px black; /* Sombra para leerse mejor */
            }
            
            /* Cajas de Máscara con TEXTO */
            .mask-red-box {
                background-color: #FF4B4B; 
                height: 100%;
                width: 25%;
                margin-left: 30%; 
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
                font-size: 0.7rem;
                box-shadow: 0 0 10px rgba(255, 75, 75, 0.5);
                z-index: 5;
            }
            
            .mask-blue-box {
                background-color: #4DA6FF;
                height: 50%; /* Ocupa la mitad vertical */
                width: 100%;
                position: absolute;
                top: 0; /* Pegado arriba */
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
                font-size: 0.7rem;
                box-shadow: 0 0 15px rgba(77, 166, 255, 0.6);
                z-index: 5;
            }
        </style>

        <div style="font-size:0.8rem; margin-bottom:2px;">1. Time Masking (Corte vertical)</div>
        <div class="signal-bg">
            <div class="mask-red-box">DATOS OCULTOS</div>
            <span class="mask-label">← Audio Original →</span>
        </div>

        <div style="font-size:0.8rem; margin-bottom:2px;">2. Frequency Masking (Corte horizontal)</div>
        <div class="signal-bg">
            <div class="mask-blue-box">DATOS OCULTOS</div>
            <span class="mask-label" style="margin-top: 30px; width: 100%; text-align: center; margin-left: 0;">
                ← Audio Original (Frecuencias Bajas) →
            </span>
        </div>
        """, unsafe_allow_html=True)


# PESTAÑA 3: ONECYCLELR (LA CORRECCIÓN IMPORTANTE)
with tab3:
    st.markdown("#### 📉 Super-Convergencia (One Cycle Policy)")
    
    c_desc, c_code = st.columns([1.3, 1])
    
    with c_desc:
        st.write("Se utilizó la estrategia de **Leslie Smith** para acelerar la convergencia en solo 5 épocas. El *Learning Rate* (LR) varía dinámicamente en 3 fases:")
        
        st.markdown("""
        1.  **Warm-up (30%):** El LR sube de `1e-4` a `1e-3`. Permite salir de mínimos locales planos iniciales.
        2.  **Annealing (40%):** El LR baja siguiendo una curva coseno (Cosine Annealing). El modelo converge rápido.
        3.  **Fine-Tuning (30%):** El LR cae drásticamente (`1e-6`) para asentarse en el mínimo global óptimo.
        """)
    
    with c_code:
        st.markdown("###### Implementación:")
        st.code("""
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,     # Pico máximo
    epochs=5,        # Ciclo corto
    steps_per_epoch=len(train_loader),
    pct_start=0.3,   # 30% subida
    div_factor=10    # Start = Max/10
)
        """, language="python")
        
    st.success("🎯 **Resultado:** Se alcanzó un 74.7% de Accuracy en tiempo récord, evitando estancamientos en las primeras épocas.")

st.markdown("---")

# --- SECCIÓN 4: RESULTADOS REALES ---
st.header("4. Resultados Finales")

# Aquí ponemos TUS DATOS REALES
m1, m2, m3 = st.columns(3)
m1.metric("Precisión PANNs", "60.14%", "Baseline")
m2.metric("Precisión PaSST", "74.70%", "+14.5% Mejora")
m3.metric("Gap Generalización", "< 6%", "Sin Overfitting")

st.markdown("""
<div style='background-color: #1a1c24; padding: 20px; border-radius: 10px; border: 1px solid #333; margin-top: 20px; font-style: italic; color: #ccc;'>
"Conclusión: Aunque las CNNs (PANNs) son eficientes, la capacidad de los Transformers (PaSST) para modelar dependencias a largo plazo resulta crítica en bioacústica, logrando una mejora sustancial del 14% en precisión a cambio de un mayor coste computacional."
</div>
""", unsafe_allow_html=True)