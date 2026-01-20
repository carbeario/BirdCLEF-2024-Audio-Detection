TFG_BirdCLEF_2024/
│
├── home.py                    # Portada del proyecto
├── requirements.txt           # Librerías necesarias (streamlit, torch, librosa...)
├── packages.txt               # (Opcional) Si necesitas instalar ffmpeg en el servidor
│
├── pages/                     # Pantallas del proyecto
│   ├── 1_Detector.py          # La funcionalidad principal (Subir audio -> Predicción)
│   ├── 2_catalogue.py         # Buscador de especies y fichas técnicas
│   └── 3_Methodology.py       # Explicación del modelo, arquitectura y métricas
│
├── src/                       #Lógica del proyecto
│   ├── __init__.py
│   ├── config.py              # Constantes (SAMPLE_RATE, N_MELS, LISTA_ESPECIES...)
│   ├── audio_utils.py         # Funciones de troceado (sliding window), librosa, pydub
│   └── inference.py           # Función que carga modelo y orquesta la predicción
│
├── assets/                    # Recursos estáticos
│   ├── logo.png               # Logo del proyecto
│   ├── styles.css             # Si quieres inyectar CSS personalizado
│   ├── species_metadata.csv   # Base de datos estática con nombres y enlaces de las 101 especies soportadas
│   └── ejemplos/              # 2 o 3 audios .ogg de prueba para el usuario
│
└── weights/                   # Donde se guarda el archivos .pth entrenado
    └── panns_v1.pth