# BirdCLEF 2024: Advanced Bioacoustic Detection System

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-EE4C2C?logo=pytorch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Microservices-2496ED?logo=docker&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41-FF4B4B?logo=streamlit&logoColor=white)

> **Sistema End-to-End contenerizado mediante arquitectura de microservicios para la clasificación de especies aviares con Deep Learning.**

## Sobre el Proyecto
Este proyecto soluciona el desafío de identificar especies en entornos ruidosos (Western Ghats, India) mediante el análisis de espectrogramas Mel. A diferencia de scripts académicos aislados, esta solución está diseñada como una arquitectura de **microservicios orientada a producción**, resolviendo problemas críticos de *timeout* en inferencia pesada (32kHz) mediante desacoplamiento.

## 🏗️ Arquitectura del Sistema (Desacoplado)
Por requisitos de evaluación del TFG, el código reside en un repositorio único (monorepo), pero el sistema opera estrictamente bajo dos contenedores aislados orquestados por Docker Compose:

* **📦 Contenedor UI (Frontend - `Dockerfile.ui`):** Gestionado por `home.py` y `pages/` (Streamlit). Implementa un patrón de *Polling Health Check* para evitar caídas asíncronas.
* **🧠 Contenedor API (Backend - `Dockerfile.api`):** Servidor FastAPI asíncrono en la carpeta `api/` y `src/`. Carga modelos PANNs y PaSST directamente en VRAM/RAM y gestiona el pipeline DSP.

## Despliegue Rápido (Docker Compose)
El sistema está diseñado para levantar toda la infraestructura de red, dependencias y modelos preentrenados con un solo comando.

1.  **Clonar repositorio:**
    ```bash
    git clone [https://github.com/carbeario/BirdCLEF-2024-Audio-Detection.git](https://github.com/carbeario/BirdCLEF-2024-Audio-Detection.git) 
    cd BirdCLEF-2024-Audio-Detection
    ```

2.  **Orquestar Microservicios:**
    ```bash
    docker-compose up --build
    ```
    * El Frontend interactivo estará accesible en: `http://localhost:8501`
    * La documentación interactiva de la API (Swagger UI) en: `http://localhost:8000/docs`