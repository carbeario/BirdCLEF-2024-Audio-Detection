FROM python:3.10-slim

# 1. Instalar herramientas de sistema (Motor)
RUN apt-get update && apt-get install -y \
    build-essential libsndfile1 ffmpeg graphviz curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Instalar librerías Python (Combustible)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3. Descargar los Cerebros (Modelos)
RUN python3 -c "from huggingface_hub import hf_hub_download; \
    repo = 'CarLSBR/BirdCLEF-2024-PANNs'; \
    files = ['panns_mejor.pth', 'passt_mejor.pth', 'Cnn14_mAP=0.431.pth']; \
    [hf_hub_download(repo_id=repo, filename=f, local_dir='weights', local_dir_use_symlinks=False) for f in files]"

# 4. Copiar la App (Carrocería)
COPY . .

# 5. Configurar dependencias legacy (Ajuste de motor)
# Esto no es un parche, es poner el archivo de configuración donde la librería lo lee.
RUN mkdir -p /root/panns_data && \
    cp assets/class_labels_indices.csv /root/panns_data/class_labels_indices.csv

EXPOSE 8501
CMD ["streamlit", "run", "home.py", "--server.port=8501", "--server.address=0.0.0.0"]