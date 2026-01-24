# BirdCLEF 2024: Advanced Bioacoustic Detection System

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-EE4C2C?logo=pytorch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Production_Ready-2496ED?logo=docker&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41-FF4B4B?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success)

> **Sistema End-to-End contenerizado para la clasificación de especies aviares mediante arquitecturas híbridas de Deep Learning.**


## Sobre el Proyecto
Este proyecto soluciona el desafío de identificar especies de aves en entornos ruidosos mediante el análisis de espectrogramas de audio. A diferencia de scripts académicos aislados, esta solución está diseñada como un **microservicio de producción**.

**Características Principales:**
- **Arquitectura Híbrida:** Implementación de modelos **CNN14 (PANNs)** y **Transformers (PaSST)** para maximizar la robustez.
- **Ingeniería de MLOps:** Gestión de pesos desacoplada mediante **Hugging Face Hub** y CI/CD friendly.
- **Despliegue Agnóstico:** Empaquetado en **Docker** para garantizar la reproducibilidad exacta en cualquier entorno (Linux, Windows, Cloud).
- **Interfaz Interactiva:** Dashboard analítico desarrollado en Streamlit para usuarios no técnicos.

## Tech Stack
* **Core:** Python 3.10
* **Deep Learning:** PyTorch, Torchaudio, Panns-Inference, Hear21Passt
* **Procesamiento de Señal:** Librosa, Soundfile
* **Infraestructura:** Docker (Multi-stage build strategy)
* **Visualización:** Streamlit, Matplotlib

## Despliegue Rápido (Docker)
El sistema está diseñado para ser "Plug & Play". No requiere instalación manual de librerías.

1.  **Clonar repositorio:**
    ```bash
    git clone [https://github.com/TU_USUARIO/TFG_BirdCLEF_2024.git](https://github.com/TU_USUARIO/TFG_BirdCLEF_2024.git)
    cd TFG_BirdCLEF_2024
    ```

2.  **Construir Contenedor:**
    *Nota: Este paso descarga automáticamente los modelos optimizados desde Hugging Face.*
    ```bash
    docker build -t birdclef-app .
    ```

3.  **Ejecutar Servicio:**
    ```bash
    docker run -p 8501:8501 birdclef-app
    ```
    Accede a la aplicación en: `http://localhost:8501`

## Estructura del Proyecto
Para detalles técnicos sobre la organización de ficheros y decisiones de arquitectura, consultar [DOCUMENTATION.md](./Documentation.md).

---
*Desarrollado por Carlos Beato Rioja como Trabajo de Fin de Grado (TFG) - 2026*