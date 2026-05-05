import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import librosa 
import base64
import io

# Importamos constantes de config
from .config import SAMPLE_RATE, DURATION, FMIN, FMAX, HOP_LENGTH

def process_audio(audio_path, n_mels=64):
    """
    Procesa el audio exactamente igual que en el entrenamiento.
    Args:
        audio_path: Ruta al archivo.
        n_mels: 64 para PANNs, 128 para PaSST.
    Returns:
        Tensor de forma [1, 1, n_mels, time] listo para la IA.
    """
    try:
        # 1. Cargar Audio con Torchaudio 
        sig, sr = torchaudio.load(audio_path)
        
        # 2. Convertir a Mono 
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
            sig = sig[:, :target_len]
            
        # 5. Generar Espectrograma (MelSpectrogram)
        # Aquí usamos n_mels dinámico
        mel_transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_mels=n_mels, 
            n_fft=1024,     
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
        print(f"Error crítico procesando audio: {e}")
        return None

def generar_espectrograma_visual(audio_path):
    """
    Genera una gráfica para mostrar en la web (Streamlit).
    """
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        # Recortar a 10 segundos máx para visualización
        if len(y) > SAMPLE_RATE * 10:
            y = y[:SAMPLE_RATE*10]
            
        # Espectrograma lineal estándar para visualización humana
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        
        fig, ax = plt.subplots(figsize=(10, 3))
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
    


plt.switch_backend('Agg')

def get_spectrogram_image(audio_path, n_mels=64):
    """
    Genera una representación visual del Mel-Spectrogram.
    Devuelve: String en Base64 listo para enviar por la API.
    """
    try:
        # 1. Cargar audio 
        y, sr = librosa.load(audio_path, sr=32000)
        
        # 2. Calcular Mel-Spectrogram
        # Usamos los parámetros estándar de BirdCLEF (fmin=50, fmax=14000)
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, 
            fmin=50, fmax=14000, n_fft=1024, hop_length=320
        )
        S_dB = librosa.power_to_db(S, ref=np.max)

        # 3. Dibujar con Matplotlib
        plt.figure(figsize=(10, 3))
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmin=50, fmax=14000, cmap='magma')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel-Spectrogram ({n_mels} mels)')
        plt.tight_layout()

        # 4. Guardar en memoria (BytesIO) en lugar de disco
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close() 
        
        # 5. Codificar a Base64
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{image_base64}"

    except Exception as e:
        print(f"Error generando espectrograma: {e}")
        return None