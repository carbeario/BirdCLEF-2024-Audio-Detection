# 1. IMAGEN BASE
FROM python:3.10-slim

# 2. VARIABLES DE ENTORNO
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. DEPENDENCIAS DE SISTEMA
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. DIRECTORIO DE TRABAJO
WORKDIR /app



# 5. INSTALACIÓN DE LIBRERÍAS
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Copiamos carpeta local 'panns_data' a la ruta de root del contenedor
COPY panns_data /root/panns_data


# 6. COPIA DEL PROYECTO (Sin los pesos, el .gitignore se encarga)
COPY . .


# 7. PRE-DESCARGA DESDE HUGGING FACE
# Bajamos el archivo con su nombre real y lo dejamos en 'weights/'
RUN python3 -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='CarLSBR/BirdCLEF-2024-PANNs', \
    filename='panns_inference.pth', \
    local_dir='weights', \
    local_dir_use_symlinks=False)"

# 8. PUERTO Y HEALTHCHECK
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 9. COMANDO DE ARRANQUE
CMD ["streamlit", "run", "home.py", "--server.port=8501", "--server.address=0.0.0.0"]