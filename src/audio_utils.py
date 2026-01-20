import librosa
import librosa.display
import numpy as np
import torch
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN (Sincronizada con Entrenamiento) ---
SAMPLE_RATE = 32000
DURATION = 5 
N_MELS = 64
N_FFT = 2048      
HOP_LENGTH = 512  

def procesar_audio(audio_path):
    """
    Estrategia de VENTANA DESLIZANTE (Sliding Window):
    En lugar de un solo recorte, generamos múltiples recortes de 5 segundos
    que cubren todo el audio.
    Retorna: Tensor de forma (N_fragmentos, 3, 64, 313)
    """
    try:
        # 1. Cargar audio
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        
        # Eliminamos silencios extremos al principio/final para limpiar
        y, _ = librosa.effects.trim(y, top_db=20)

        # Si es muy corto, lo rellenamos una vez
        target_len = SAMPLE_RATE * DURATION
        if len(y) < target_len:
            padding = target_len - len(y)
            y = np.pad(y, (0, padding), mode='constant')

        # 2. Generar ventanas deslizantes
        # Avanzamos de 2.5 en 2.5 segundos (50% de solapamiento) para no perder nada
        stride = int(target_len * 0.5) 
        chunks = []
        
        # Si el audio es muy largo, limitamos a los primeros 60 seg para no saturar memoria RAM
        limit_samples = SAMPLE_RATE * 60 
        y = y[:limit_samples]

        for start in range(0, len(y) - target_len + 1, stride):
            end = start + target_len
            chunk = y[start:end]
            
            # --- PROCESAMIENTO POR FRAGMENTO ---
            melspec = librosa.feature.melspectrogram(
                y=chunk, sr=sr, n_mels=N_MELS, fmax=14000, 
                n_fft=N_FFT, hop_length=HOP_LENGTH
            )
            logmelspec = librosa.power_to_db(melspec, ref=np.max)
            
            # Normalización Min-Max (Igual que entrenamiento)
            min_val = np.min(logmelspec)
            max_val = np.max(logmelspec)
            logmelspec = (logmelspec - min_val) / (max_val - min_val + 1e-6)
            
            # Stack 3 canales
            img = np.stack([logmelspec]*3, axis=-1)
            chunks.append(img)

        # Si no salieron chunks (caso raro), forzamos uno
        if not chunks:
            # Repetir lógica de un solo chunk
            return None 

        # 3. Convertir lista de chunks a Tensor Batch
        # De lista [(Mel, Time, 3), ...] a numpy (N, Mel, Time, 3)
        batch_np = np.array(chunks)
        
        # A Tensor PyTorch y Permutar a (N, 3, Mel, Time)
        tensor_batch = torch.tensor(batch_np, dtype=torch.float32).permute(0, 3, 1, 2)
        
        return tensor_batch
        
    except Exception as e:
        print(f"Error procesando audio: {e}")
        return None

def generar_espectrograma_visual(audio_path):
    # Función para visualizar (Mantenemos igual pero con recorte simple para la foto)
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        # Mostramos hasta 10 segundos
        if len(y) > SAMPLE_RATE * 10: 
            y = y[:SAMPLE_RATE*10]
        
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        
        fig, ax = plt.subplots(figsize=(10, 3))
        fig.patch.set_facecolor('#1a1c24') 
        ax.set_facecolor('#1a1c24')
        ax.tick_params(axis='x', colors='#888888')
        ax.tick_params(axis='y', colors='#888888')
        for spine in ax.spines.values():
            spine.set_color('#333333')
        
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax, cmap='magma')
        plt.tight_layout()
        return fig
    except Exception:
        return None