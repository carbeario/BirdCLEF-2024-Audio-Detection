# BirdCLEF 2024: Detector de Especies por Bioacústica

Este proyecto es un sistema End-to-End para la detección y clasificación de aves mediante audios, utilizando **Deep Learning (PyTorch)** y **Procesamiento de Señales Digitales**.

## 🏗 Arquitectura del Proyecto
El sistema sigue una arquitectura modular de ingeniería de software:
- `src/audio_utils.py`: Pipeline de preprocesamiento de audio (Sliding Windows, Mel-Spectrograms).
- `src/inference.py`: Motor de inferencia utilizando modelos PANNs pre-entrenados.
- `pages/`: Interfaz de usuario construida con **Streamlit**.
- `Docker`: Contenerización lista para despliegue en nube.

## 🚀 Tecnologías
- **Lenguaje:** Python 3.9
- **IA:** PyTorch, Librosa
- **Infraestructura:** Docker, Streamlit

## 🔧 Instalación y Uso
1. Clonar el repositorio.
2. Construir la imagen Docker: `docker build -t birdclef-app .`
3. Ejecutar: `docker run -p 8501:8501 birdclef-app`