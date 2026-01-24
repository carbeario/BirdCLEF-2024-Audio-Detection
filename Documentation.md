# Documentación Técnica y Estructura

Este documento detalla la arquitectura de software y la organización de ficheros del sistema BirdCLEF 2024.

## Árbol de Directorios

El proyecto sigue una estructura modular para separar la lógica de inferencia, la interfaz de usuario y los recursos estáticos.

```text
TFG_BirdCLEF_2024/
│
├── Infraestructura
│   ├── Dockerfile                 # Definición de la imagen (Python 3.10-slim)
│   ├── .dockerignore              # Optimización del contexto de build
│   ├── requirements.txt           # Dependencias Python (Versiones pineadas para estabilidad)
│   └── packages.txt               # Dependencias de sistema (OS Level)
│
├── Source Code (src/)
│   ├── config.py                  # Gestión centralizada de configuración y rutas
│   ├── inference.py               # Motor de inferencia y gestión de descargas de modelos
│   ├── audio_utils.py             # Pipeline de preprocesamiento (Audio -> Mel Spectrogram)
│   └── __init__.py
│
├── Interfaz (Streamlit)
│   ├── home.py                    # Punto de entrada de la aplicación
│   └── pages/
│       ├── 1_Detector.py          # Módulo de inferencia en tiempo real
│       ├── 2_Catalogue.py         # Base de conocimientos de especies
│       └── 3_Methodology.py       # Documentación integrada en la app
│
├── Assets & Recursos
│   ├── assets/
│   │   ├── class_labels_indices.csv # Mapeo de clases (Requisito PANNs)
│   │   ├── species_metadata.csv     # Metadatos enriquecidos
│   │   └── logo.png
│   └── weights/                   # (Generado dinámicamente) Almacén de modelos .pth
│
└── Legacy & Experiments
    └── experiments/               # Scripts de entrenamiento y validación previos