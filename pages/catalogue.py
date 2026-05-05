import streamlit as st
import math
from src import config
from src import ui_utils
from pathlib import Path

# 1. CONFIGURACIÓN
st.set_page_config(page_title="Catálogo", page_icon="🐦", layout="wide")

st.markdown("""
    <style>
        div[data-testid="column"] {
            width: fit-content !important;
            flex: 1 1 300px !important;
            min-width: 300px !important;
        }
        
        .bird-card {
            background-color: #262730;
            padding: 15px;
            border-radius: 12px;
            border: 1px solid #3b3d45;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            text-align: center;
            height: 100%;
            transition: transform 0.2s;
        }
        .bird-card:hover {
            transform: translateY(-5px);
            border-color: #4da9ff;
        }
        
        div[data-testid="stImage"] img {
            border-radius: 8px;
            object-fit: cover;
            height: 200px !important;
            width: 100% !important;
        }
    </style>
""", unsafe_allow_html=True)

# 2. GESTIÓN DE DATOS
if config.SPECIES_DF.empty:
    st.error("No se cargaron las especies.")
    st.stop()

# Header
st.title("Inventario de Especies")
c_search, c_stats = st.columns([3, 1], vertical_alignment="bottom")
with c_search:
    query = st.text_input("🔍 Buscar:", placeholder="Escribe un nombre...", label_visibility="collapsed")
with c_stats:
    st.caption(f"**{len(config.SPECIES_DF)}** especies indexadas.")

# 3. FILTRADO
if query:
    mask = (
        config.SPECIES_DF['common_name'].str.contains(query, case=False) | 
        config.SPECIES_DF['scientific_name'].str.contains(query, case=False) |
        config.SPECIES_DF['primary_label'].str.contains(query, case=False)
    )
    df_filtered = config.SPECIES_DF[mask]
else:
    df_filtered = config.SPECIES_DF

# 4. PAGINACIÓN
ITEMS_PER_PAGE = 12 
total_pages = math.ceil(len(df_filtered) / ITEMS_PER_PAGE)

if "page" not in st.session_state: st.session_state.page = 1
if query != st.session_state.get("last_query_cat", ""):
    st.session_state.page = 1
    st.session_state.last_query_cat = query

# Controles de Paginación
if total_pages > 1:
    c1, c2, c3 = st.columns([1, 4, 1])
    with c1:
        if st.button("Anterior", disabled=st.session_state.page == 1):
            st.session_state.page -= 1
            st.rerun()
    with c2:
        st.markdown(f"<div style='text-align: center; padding-top: 5px;'>Página <b>{st.session_state.page}</b> de {total_pages}</div>", unsafe_allow_html=True)
    with c3:
        if st.button("Siguiente", disabled=st.session_state.page == total_pages):
            st.session_state.page += 1
            st.rerun()

# 5. RENDERIZADO (Arquitectura Local SOTA)
st.divider()

if df_filtered.empty:
    st.warning("No hay resultados.")
else:
    start = (st.session_state.page - 1) * ITEMS_PER_PAGE
    end = start + ITEMS_PER_PAGE
    batch = df_filtered.iloc[start:end]
    cols = st.columns(4) 
    
    for i, (_, row) in enumerate(batch.iterrows()):
        with cols[i % 4]:
            
            # Datos
            nombre = row['common_name']
            cientifico = row['scientific_name']
            codigo = row['primary_label']
            
            # 🔴 EL CAMBIO CLAVE: Leer la foto desde el disco duro local
            local_photo_path = config.ASSETS_DIR / "images" / codigo / "photo.jpg"
            fallback_logo = config.ASSETS_DIR / "logo.png"
            
            if local_photo_path.exists():
                imagen_mostrar = str(local_photo_path)
            elif fallback_logo.exists():
                imagen_mostrar = str(fallback_logo)
            else:
                imagen_mostrar = "https://via.placeholder.com/300x200?text=Sin+Imagen"
            
            # Generar enlaces externos sin hacer peticiones web
            enlaces = ui_utils.get_bird_links(cientifico, codigo)
            
            # TARJETA
            with st.container(border=True):
                st.image(imagen_mostrar, use_container_width=True)
                
                # Textos
                st.subheader(nombre)
                st.markdown(f"*{cientifico}*")
                st.caption(f"ID: `{codigo}`")
                
                # Botones
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if row['url_xenocanto']:
                        st.link_button("🎵 Audio", row['url_xenocanto'], use_container_width=True)
                with col_btn2:
                    st.link_button("📖 Wiki", enlaces["wikipedia"], use_container_width=True)

st.caption("---")