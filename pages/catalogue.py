import streamlit as st
import requests
import re
from src import config 

# 1. CARGA DE ESTILOS GLOBALES
config.aplicar_estilos()
st.set_page_config(page_title="Ficha Técnica", page_icon="📖", layout="wide")

# ==========================================
# 2. CSS ESPECÍFICO (FIX DE CONTRASTE)
# ==========================================
st.markdown("""
    <style>
        /* A. ARREGLAR MÉTRICAS (Fondo oscuro y texto blanco) */
        div[data-testid="stMetric"] {
            background-color: #1a1c24; /* Fondo casi negro */
            border: 1px solid #333;
            padding: 10px !important;
            border-radius: 8px;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.1rem !important;
            color: #ffffff !important; /* Blanco puro */
            word-wrap: break-word; 
            white-space: normal;
            font-weight: 600;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.85rem !important;
            color: #a0a0a0 !important; /* Gris claro */
            margin-bottom: 5px;
        }

        /* B. CAJA DE DESCRIPCIÓN PERSONALIZADA (Reemplaza al st.info azul chillón) */
        .desc-box {
            background-color: #13151b;
            border-left: 3px solid #FF9F33; /* Línea naranja elegante */
            padding: 15px;
            border-radius: 0 5px 5px 0;
            color: #e0e0e0;
            font-size: 0.95rem;
            line-height: 1.5;
            margin-top: 10px;
        }

        /* C. IMAGEN CON ESTILO */
        img {
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        }

        /* D. TAG DE CÓDIGO (ID Modelo) */
        .id-tag {
            background-color: #2b1d1d;
            border: 1px solid #5c2b2b;
            color: #ff6b6b;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 0.9em;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. LÓGICA DE EXTRACCIÓN (WIKIPEDIA API)
# ==========================================
@st.cache_data(ttl=3600*24, show_spinner=False)
def get_bird_details(scientific_name):
    result = {
        "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/No_image_available.svg/600px-No_image_available.svg.png",
        "extract": "Descripción no disponible.",
        "familia": "No detectada",
        "status": "No evaluado ❓"
    }
    
    query = scientific_name.replace(" ", "_")
    headers = {'User-Agent': 'BirdCLEF-TFG/1.0'}

    # --- API REST (Imagen y Texto) ---
    try:
        api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query}"
        response = requests.get(api_url, headers=headers, timeout=3)
        if response.status_code == 200:
            data = response.json()
            if 'originalimage' in data:
                result["img"] = data['originalimage']['source']
            elif 'thumbnail' in data:
                result["img"] = data['thumbnail']['source']
            if 'extract' in data:
                result["extract"] = data['extract']
    except:
        pass

    # --- SCRAPING HTML (Familia y Status) ---
    html_text = ""
    try:
        html_url = f"https://en.wikipedia.org/wiki/{query}"
        html_resp = requests.get(html_url, headers=headers, timeout=5)
        if html_resp.status_code == 200:
            html_text = html_resp.text
    except:
        pass

    # --- ANÁLISIS DE TEXTO ---
    texto_busqueda = (html_text + " " + result["extract"])

    # A. FAMILIA
    match_familia = re.search(r'Family:.*?([A-Z][a-z]+idae)', texto_busqueda, re.IGNORECASE)
    if match_familia:
        result["familia"] = match_familia.group(1)
    else:
        match_simple = re.search(r'\b([A-Z][a-z]+idae)\b', texto_busqueda)
        if match_simple:
            result["familia"] = match_simple.group(1)

    # B. ESTADO UICN
    mapa_uicn = {
        "Least Concern": "Preocupación Menor (LC) 🟢",
        "Near Threatened": "Casi Amenazada (NT) 🟡",
        "Vulnerable": "Vulnerable (VU) 🟠",
        "Endangered": "En Peligro (EN) 🔴",
        "Critically Endangered": "Crítico (CR) 🚨", 
        "Extinct in the Wild": "Extinto en Silvestre (EW) 🏚️",
        "Extinct": "Extinto (EX) ⚫",
    }
    
    for eng, esp in mapa_uicn.items():
        if re.search(rf'\b{eng}\b', texto_busqueda, re.IGNORECASE):
            result["status"] = esp
            break

    return result

# ==========================================
# 4. INTERFAZ (FRONTEND)
# ==========================================
df_especies = config.SPECIES_DF

if df_especies.empty:
    st.error("No se han podido cargar los metadatos de las especies.")
    st.stop()

with st.sidebar:
    st.title(f"Catálogo")
    st.info(f"Base de datos: **{len(df_especies)} especies**")
    
    opciones = [f"{row['common_name']} ({row['primary_label']})" for _, row in df_especies.iterrows()]
    seleccion_usuario = st.selectbox("🔍 Buscar Especie:", options=opciones, index=0)
    codigo = seleccion_usuario.split("(")[-1].replace(")", "")

try:
    row = df_especies[df_especies['primary_label'] == codigo].iloc[0]
    nombre_cientifico = row['scientific_name']
    nombre_comun = row['common_name']
    
    with st.status("🔍 Consultando bases de datos...", expanded=True) as status:
        st.write("Conectando con Wikipedia API...")
        wiki_data = get_bird_details(nombre_cientifico)
        status.update(label="¡Ficha cargada!", state="complete", expanded=False)
    
    # --- CABECERA ---
    c_title, c_links = st.columns([3, 1], vertical_alignment="center")
    with c_title:
        st.markdown(f"<h1 style='margin-bottom:0px;'>{nombre_comun}</h1>", unsafe_allow_html=True)
        # Usamos la clase CSS .id-tag que definimos arriba
        st.markdown(f"""
        <div style='color: #a3a8b8; margin-top: 5px;'>
            <i>{nombre_cientifico}</i> &nbsp;•&nbsp; ID: <span class='id-tag'>{codigo}</span>
        </div>
        """, unsafe_allow_html=True)

    with c_links:
        r1, r2 = st.columns(2)
        r1.link_button("🔊 Audio", row['url_xenocanto'], use_container_width=True)
        r2.link_button("📖 Wiki", f"https://en.wikipedia.org/wiki/{nombre_cientifico.replace(' ', '_')}", use_container_width=True)

    st.markdown("---")

    # --- CUERPO ---
    col_izq, col_der = st.columns([1, 1.5], gap="large")

    with col_izq:
        st.image(wiki_data["img"], use_container_width=True)

    with col_der:
        st.subheader("Ficha Técnica")
        
        # Contenedor principal con borde
        with st.container(border=True):
            # Usamos CSS para que estas métricas se vean con fondo oscuro
            k1, k2 = st.columns([1, 1.2]) 
            k1.metric("Familia Taxonómica", wiki_data['familia'])
            k2.metric("Estado UICN", wiki_data['status'])

            st.markdown("<div style='margin-top: 20px; margin-bottom: 5px; font-weight: bold; color: #ccc;'>Descripción:</div>", unsafe_allow_html=True)
            
            # Usamos el div personalizado .desc-box
            if wiki_data['extract'] and wiki_data['extract'] != "Descripción no disponible.":
                 st.markdown(f'<div class="desc-box">{wiki_data["extract"]}</div>', unsafe_allow_html=True)
            else:
                 st.warning("No se encontró descripción en Wikipedia.")
            
            st.markdown(f"""
            <div style="font-size: 0.8em; opacity: 0.5; margin-top: 20px; text-align: right; border-top: 1px solid #333; padding-top: 10px;">
            ℹ️ Datos sincronizados con <b>Wikipedia Live</b>
            </div>
            """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error cargando la ficha: {e}")