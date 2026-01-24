import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import librosa # Solo lo usamos para la visualización bonita del gráfico, no para el modelo.

# Importamos constantes de config
from .config import SAMPLE_RATE, DURATION, FMIN, FMAX, HOP_LENGTH

def process_audio(audio_path, n_mels=64):
    """
    Procesa el audio EXACTAMENTE igual que en el entrenamiento.
    Args:
        audio_path: Ruta al archivo.
        n_mels: 64 para PANNs, 128 para PaSST.
    Returns:
        Tensor de forma [1, 1, n_mels, time] listo para la IA.
    """
    try:
        # 1. Cargar Audio con Torchaudio (Más rápido y preciso para PyTorch)
        sig, sr = torchaudio.load(audio_path)
        
        # 2. Convertir a Mono (Promedio de canales)
        if sig.shape[0] > 1:
            sig = torch.mean(sig, dim=0, keepdim=True)
        
        # 3. Resample (Si el usuario sube un audio a 44.1kHz, lo bajamos a 32kHz)
        if sr != SAMPLE_RATE:
            resampler = T.Resample(sr, SAMPLE_RATE)
            sig = resampler(sig)
        
        # 4. Ajuste de Longitud (Padding o Recorte a 5 segundos exactos)
        target_len = SAMPLE_RATE * DURATION
        current_len = sig.shape[1]
        
        if current_len < target_len:
            # Rellenar con ceros al final
            sig = torch.nn.functional.pad(sig, (0, target_len - current_len))
        elif current_len > target_len:
            # Cortar (Nos quedamos con los primeros 5 seg)
            # Nota: En una app v2.0 podríamos hacer ventanas deslizantes, 
            # pero para compatibilidad total con tu TFG, un recorte simple es más seguro ahora.
            sig = sig[:, :target_len]
            
        # 5. Generar Espectrograma (MelSpectrogram)
        # Aquí usamos n_mels dinámico
        mel_transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_mels=n_mels,  # <--- CLAVE: 64 o 128
            n_fft=1024,     # Coincide con tu entrenamiento
            hop_length=512,
            f_min=FMIN,
            f_max=FMAX
        )
        
        spec = mel_transform(sig)
        
        # 6. Escala Logarítmica (dB)
        db_transform = T.AmplitudeToDB(top_db=80)
        spec = db_transform(spec)
        
        # 7. Normalización (Min-Max scaling aproximado, como en el training)
        spec = (spec + 80.0) / 80.0
        
        # 8. Añadir dimensión de Batch [Batch, Channel, Mels, Time]
        # Salida: [1, 1, n_mels, 313]
        spec = spec.unsqueeze(0) 
        
        return spec

    except Exception as e:
        print(f"❌ Error crítico procesando audio: {e}")
        return None

def generar_espectrograma_visual(audio_path):
    """
    Genera una gráfica bonita para mostrar en la web (Streamlit).
    Aquí sí usamos Librosa porque sus gráficas son más fáciles de hacer.
    """
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        # Recortar a 10 segundos máx para visualización
        if len(y) > SAMPLE_RATE * 10:
            y = y[:SAMPLE_RATE*10]
            
        # Espectrograma lineal estándar para visualización humana
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        
        fig, ax = plt.subplots(figsize=(10, 3))
        # Estilo oscuro 'Cyberpunk'
        fig.patch.set_facecolor('#0e1117') 
        ax.set_facecolor('#0e1117')
        ax.tick_params(axis='x', colors='#888888')
        ax.tick_params(axis='y', colors='#888888')
        for spine in ax.spines.values():
            spine.set_color('#333333')
            
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax, cmap='inferno')
        plt.tight_layout()
        return fig
    except Exception:
        return None