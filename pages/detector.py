import streamlit as st
import requests
import os
import time
from src import config      
from src import ui_utils    

# ==========================================
# CONFIGURACIÓN DE PÁGINA
# ==========================================
st.set_page_config(page_title="Detector BirdCLEF", page_icon="🦅", layout="wide")

# ==========================================
# 🚦 HEALTH CHECK: 
# ==========================================
def esperar_api():
    """
    Bloquea la ejecución mostrando un spinner hasta que la API responda (200 OK).
    No recarga la página, solo esperaes un.
    """
    api_url_full = os.getenv("API_URL", "http://localhost:8000/predict")
    api_root = api_url_full.replace("/predict", "")
    
    def hay_conexion():
        try:
            r = requests.get(api_root, timeout=1)
            return r.status_code == 200
        except:
            return False

    if hay_conexion():
        return True
    placeholder = st.empty()
    
    with placeholder.container():
        with st.spinner("Iniciando Cerebro Digital (Cargando modelos PANNs/PaSST en RAM)..."):
            
            intentos = 0
            max_intentos = 300 
            
            while not hay_conexion():
                time.sleep(1) 
                intentos += 1
                
                if intentos >= max_intentos:
                    st.error("La API está tardando demasiado o Docker no arranca.")
                    st.stop() 
            
    placeholder.success("¡Sistemas Online!")
    time.sleep(1) 
    placeholder.empty() 
    return True

if not esperar_api():
    st.stop() 




# ==========================================
# INTERFAZ PRINCIPAL
# ==========================================

# --- SIDEBAR ---
with st.sidebar:
    st.header("Configuración IA")
    modelo_seleccionado = st.radio(
        "Selecciona el modelo:",
        ("panns", "passt"),
        format_func=lambda x: "PANNs (Rápido)" if x == "panns" else "PaSST (Preciso)"
    )
    st.divider()
    st.caption("**Nota:** PaSST analiza relaciones temporales largas, ideal para fondos ruidosos.")

# --- CABECERA ---
st.title("Detector de Aves: BirdCLEF 2024")
st.markdown("Identificación automática de especies mediante análisis de espectrogramas.")

# ==========================================
# SELECTOR DUAL (SUBIR vs BIBLIOTECA)
# ==========================================
col_radio, _ = st.columns([2, 1])
with col_radio:
    fuente = st.radio(
        "¿Origen del audio?",
        ["Subir mi propio archivo", "Seleccionar de la Biblioteca (Assets)"],
        horizontal=True
    )

archivo_bytes = None
nombre_archivo = ""
tipo_mime = ""

# --- CASO A: SUBIR ARCHIVO ---
if fuente == "Subir mi propio archivo":
    uploaded = st.file_uploader("Arrastra tu archivo aquí (.wav, .mp3)", type=["wav", "mp3", "ogg", "flac"])
    if uploaded:
        archivo_bytes = uploaded.getvalue()
        nombre_archivo = uploaded.name
        tipo_mime = uploaded.type
        st.audio(archivo_bytes, format=tipo_mime)

# --- CASO B: USAR ASSETS ---
else:
    # Busca en 'assets' o 'assets/ejemplos'
    rutas_posibles = ["assets", "assets/samples"]
    ruta_valida = None
    archivos = []

    for r in rutas_posibles:
        if os.path.exists(r):
            # Filtramos solo audios
            encontrados = [f for f in os.listdir(r) if f.lower().endswith(('.wav', '.mp3', '.ogg'))]
            if encontrados:
                ruta_valida = r
                archivos = encontrados
                break
    
    if archivos:
        seleccion = st.selectbox("Selecciona un audio de prueba:", archivos)
        ruta_completa = os.path.join(ruta_valida, seleccion)
        
        # Leer en binario
        with open(ruta_completa, "rb") as f:
            archivo_bytes = f.read()
        
        nombre_archivo = seleccion
        tipo_mime = "audio/wav" # Genérico
        
        st.info(f"Cargado desde: `{ruta_completa}`")
        st.audio(archivo_bytes, format=tipo_mime)
    else:
        st.warning("No se encontraron archivos de audio en la carpeta 'assets'.")

# ==========================================
#  MOTOR DE INFERENCIA
# ==========================================
st.divider()

if archivo_bytes:
    col_btn, col_msg = st.columns([1, 3])
    with col_btn:
        analizar = st.button("Analizar Audio", type="primary", use_container_width=True)

    if analizar:
        api_url = os.getenv("API_URL", "http://localhost:8000/predict")
        
        # Preparamos payload
        files = {"file": (nombre_archivo, archivo_bytes, tipo_mime)}
        params = {"model": modelo_seleccionado}
        
        with st.spinner(f" {modelo_seleccionado.upper()} está escuchando... (Esto puede tardar un poco)"):
            try:
                # Timeout alto (300s) para dar tiempo a PaSST si va lento
                response = requests.post(api_url, files=files, params=params, timeout=300)
                
                if response.status_code == 200:
                    data = response.json()
                    preds = data.get("predictions", [])
                    img_b64 = data.get("mel_spectrogram", None)
                    
                    st.success(" Análisis completado con éxito.")
                    
                    # 1. VISUALIZACIÓN ESPECTROGRAMA (XAI)
                    if img_b64:
                        with st.expander(" Ver lo que ve la IA (Mel-Spectrogram)", expanded=True):
                            st.image(img_b64, use_container_width=True)
                            st.caption("Representación visual de frecuencias (Eje Y) vs Tiempo (Eje X).")

                    # 2. RESULTADOS
                    st.subheader("Ranking de Especies Detectadas")
                    
                    if not preds:
                        st.warning("No se detectó ninguna especie con suficiente confianza.")
                    
                    
                    #Se limitan los resultados a una confianza del 0.30
                    preds_shown = [p for p in preds if p['score'] >= 0.30 ]
                    low_confidence=False

                    if preds_shown:
                        top_score = preds_shown[0]['score']
                        #Se elige solo los resultados más cercanos al mejor
                        preds_shown = [p for p in preds_shown if p['score'] >= (top_score - 0.10)][:3]
                    else:
                    #Si nadie ha llegado al 0.30, sacamos la mejor
                        preds_shown = [preds[0]] if preds else []
                        low_confidence = True
                    
                    for i, p in enumerate(preds_shown):
                        
                        score = p['score']
                        codigo = p['label']
                        rank = i + 1

                        # Recuperar metadatos del CSV
                        common_name = codigo
                        scientific_name = codigo
                        if not config.SPECIES_DF.empty:
                            row = config.SPECIES_DF[config.SPECIES_DF['primary_label'] == codigo]
                            if not row.empty:
                                common_name = row.iloc[0]['common_name']
                                scientific_name = row.iloc[0]['scientific_name']
                                

                        local_photo_path = config.ASSETS_DIR / "images" / codigo / "photo.jpg"

                        if local_photo_path.exists():
                            image_to_show = str(local_photo_path)
                        elif config.LOGO_PATH.exists():
                            image_to_show = str(config.LOGO_PATH)
                        
                        links = ui_utils.get_bird_links(scientific_name, ebird_code=codigo)
                        
                        
                        # Tarjeta
                        with st.container(border=True):
                            if low_confidence:
                                st.warning("**Detección incierta:** El modelo tiene dudas, pero esta es la opción más probable.")
                            c_img, c_txt = st.columns([1, 4])
                            
                            with c_img:
                                st.image(image_to_show, use_container_width=True)
                            
                            with c_txt:
                                st.markdown(f"{common_name}")
                                st.markdown(f"_{scientific_name}_ | **Confianza:** `{score:.1%}`")
                                st.progress(score)
                                
                                # Botonera de enlaces
                                c1, c2, c3 = st.columns(3)
                                c1.link_button("eBird", links['ebird'])
                                c2.link_button("Wikipedia", links['wikipedia'])
                                c3.link_button("Xeno-Canto", links['xeno-canto'])

                else:
                    st.error(f" Error del Servidor: {response.status_code}")
                    with st.expander("Ver detalles técnicos"):
                        st.write(response.text)

            except requests.exceptions.Timeout:
                st.error("El modelo tardó demasiado en responder. Inténtalo de nuevo (ya estará cargado en RAM).")
            except requests.exceptions.ConnectionError:
                st.error("No se pudo conectar a la API. Asegúrate de que Docker está corriendo.")
            except Exception as e:
                st.error(f"Error inesperado: {e}")

else:
    st.info("Selecciona una fuente de audio para comenzar.")