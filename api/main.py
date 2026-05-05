from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import shutil
import os
import uuid
from src import inference
from src import audio_utils
from enum import Enum
from pydantic import BaseModel
from typing import List, Dict, Any

class ModelOptions(str, Enum):
    panns = "panns"
    passt = "passt"


class PredictionResponse(BaseModel):
    model_used: str
    predictions: List[Dict[str, Any]] # O la estructura exacta que devuelva decode_top3
    mel_spectrogram: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model_used": "panns",
                    "predictions": [
                        {"species": "Turdus merula", "probability": 0.94},
                        {"species": "Erithacus rubecula", "probability": 0.04},
                        {"species": "Parus major", "probability": 0.01}
                    ],
                    "mel_spectrogram": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
                }
            ]
        }
    }




# ==========================================
#   Carga al inicio (Eager Loading)
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("INICIANDO API: Preparando modelos...")
    
    # Directorio temporal para los audios
    os.makedirs("temp_uploads", exist_ok=True)
    
    try:
        # Usamos función load_model para pre-cargar en RAM
        # Esto activará el @lru_cache del código
        print("Cargando PANNs (y descargando si hace falta)...")
        inference.load_model("panns") 
        print("PANNs listo.")

        print("Cargando PaSST...")
        inference.load_model("passt")
        print("PaSST listo.")
        
        print("¡SISTEMA LISTO!")
        
    except Exception as e:
        print(f"Advertencia en carga inicial: {e}")    
    yield
    
    # Limpieza al cerrar
    print("Limpiando archivos temporales...")
    if os.path.exists("temp_uploads"):
        shutil.rmtree("temp_uploads")

# ==========================================
# DEFINICIÓN DE LA APP
# ==========================================
app = FastAPI(title="BirdCLEF API",
                description="API para la clasificación de audios usando modelos neuronales (PANNs y PaSST).",
                version="2.0.0",
                lifespan=lifespan)



@app.get("/")
def read_root():
    return {"status": "online", "message": "BirdCLEF API ready!"}


@app.post("/predict", summary="Clasificar audio de ave", tags=["Inferencia"],response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), model: ModelOptions = ModelOptions.panns):
    """
    Sube un archivo de audio para detectar la especie de ave.
    
    - **file**: Archivo de audio (soportados: .wav, .mp3, .ogg, .flac).
    - **model**: Modelo a utilizar para la inferencia (selecciona del desplegable).
    
    **Devuelve:** Un JSON con el modelo usado, el top 3 de predicciones y el espectrograma Mel en base64.
    """
    
    
    # 1. Validar extensión
    if not file.filename.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
        raise HTTPException(status_code=415, detail="Formato no soportado")
    
    # 2. Guardar archivo temporalmente
    temp_filename = f"temp_uploads/{uuid.uuid4()}_{file.filename}"
    
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 3. Generar Espectrograma (Para la UI)
        with open(temp_filename, "rb") as f:
            spectrogram_base64 = audio_utils.get_spectrogram_image(f)
        
        # 4. Inferencia (Usando TU código original)
        raw_probs = inference.predict_bird(temp_filename, model_key=model)
        
        if raw_probs is None:
            raise HTTPException(status_code=500, detail="Error interno en el modelo (retornó None)")
            
        # 5. Decodificar 
        results = inference.decode_top3(raw_probs)
        
        # 6. Responder
        return JSONResponse(content={
            "model_used": model,
            "predictions": results,
            "mel_spectrogram": spectrogram_base64
        })

    except Exception as e:
        print(f"Error en endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # 7. Limpieza: Borrar el archivo temporal 
        if os.path.exists(temp_filename):
            os.remove(temp_filename)