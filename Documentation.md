# Documentación Técnica y Estructura

Este documento detalla la arquitectura de software y la organización de ficheros del sistema BirdCLEF 2024.

## Árbol de Directorios

El proyecto sigue una estructura modular y desacoplada adaptada a un entorno de microservicios orquestado por Docker Compose. Aunque el código reside en un único repositorio para cumplir con los requisitos de evaluación académica, la separación lógica entre componentes es absoluta.

```text
TFG_BirdCLEF_2024/
│
├── api/                           # Microservicio Backend (Servidor FastAPI asíncrono)
│   └── main.py                    # Endpoints de la API RESTful (?model=panns/passt)
│
├── src/                           # Núcleo Computacional y Algorítmico (Inferencia & DSP)
│   ├── config.py                  # Centralización de parámetros globales y rutas
│   ├── inference.py               # Motor de inferencia (Patrón Singleton y descarga HF)
│   └── audio_utils.py             # Pipeline DSP (Audio -> Remuestreo 32kHz -> Espectrograma Log-Mel)
│
├── pages/ & home.py               # Capa de Presentación (Frontend ligero en Streamlit)
│   ├── home.py                    # Punto de entrada y cuadro de mando estadístico (Dashboard)
│   └── pages/
│       ├── 1_Detector.py          # Módulo de inferencia en tiempo real con Polling Health Check
│       ├── 2_Catalogue.py         # Base de conocimientos local-first con imágenes persistidas
│       └── 3_Methodology.py       # Documentación técnica integrada e IA Explicable (XAI)
│
├── assets/                        # Recursos Estáticos y Datos Tabulares
│   ├── class_labels_indices.csv   # Mapeo de índices latentes (Requisito PANNs)
│   └── species_metadata.csv       # Registro de metadatos taxonómicos enriquecidos
│
├── experiments/                   # Módulo de Análisis Exploratorio (EDA) y scripts previos
│   └── decide.py                  # Algoritmo de selección de clases mediante el Método del Codo
│
├── weights/                       # Almacén protegido de pesos binarios (.pth) [Ignorado en Git]
│
└── Archivos de Infraestructura y Orquestación (Raíz)
    ├── docker-compose.yml         # Orquestador del ecosistema y definición de la red privada "birdnet"
    ├── Dockerfile.api             # Definición del contenedor de IA (Backend - Python 3.10-slim + FFmpeg)
    ├── Dockerfile.ui              # Definición del contenedor de presentación (Frontend - Streamlit ligero)
    ├── requirements-api.txt       # Dependencias exclusivas del backend (PyTorch, FastAPI, etc.)
    └── requirements-ui.txt        # Dependencias exclusivas del frontend (Streamlit, Pandas, etc.)