import torch
import torch.nn as nn
import torch.nn.functional as F
import types
import os
import numpy as np
import sys
from functools import lru_cache
from huggingface_hub import hf_hub_download

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Importación de PaSST
try:
    from hear21passt.base import get_basic_model
except ImportError:
    pass

from panns_inference import AudioTagging
from src import config
from src.audio_utils import process_audio

HF_REPO_ID = "CarLSBR/BirdCLEF-2024-PANNs" 

# --- FUNCIONES AUXILIARES DE PANNS (Intactas) ---
def forward_panns_fix(self, x, mixup_lambda=None):
    x = x[:, 0:1, :, :]
    x = x.transpose(2, 3)
    x = x.transpose(1, 3)
    x = self.bn0(x)
    x = x.transpose(1, 3)
    x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
    x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
    x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
    x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
    x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
    x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
    x = torch.mean(x, dim=3)
    (x1, _) = torch.max(x, dim=2)
    x2 = torch.mean(x, dim=2)
    x = x1 + x2
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.fc1(x)
    x = F.relu(x)
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.fc_audit(x)
    return x

def ensure_model_exists(model_key):
    default_paths = {
        'panns': 'weights/panns_v1.pth',
        'panns_base': 'weights/Cnn14_mAP=0.431.pth',
        'passt': 'weights/passt_v1.pth'
    }
    path_dict = getattr(config, 'MODEL_PATHS', default_paths)
    local_path = path_dict.get(model_key, f"weights/{model_key}.pth")

    if os.path.exists(local_path): return local_path
    
    filename = os.path.basename(local_path)
    folder = os.path.dirname(local_path)
    if not os.path.exists(folder): os.makedirs(folder, exist_ok=True)
    
    print(f"[SYSTEM] Descargando {filename} desde Hugging Face...")
    try:
        hf_hub_download(repo_id=HF_REPO_ID, filename=filename, local_dir=folder, local_dir_use_symlinks=False)
        return local_path
    except Exception as e:
        print(f"Error descargando modelo: {e}")
        return None

# --- CARGA CON CACHÉ ---
@lru_cache(maxsize=None)
def load_model(model_key):
    device = getattr(config, 'DEVICE', 'cpu')
    num_classes = len(getattr(config, 'SPECIES_LIST', range(101)))
    path_pesos = ensure_model_exists(model_key)
    if not path_pesos: return None, 64

    print(f"--> CARGANDO: {model_key.upper()}...")
    try:
        if model_key == 'panns':
            path_base = ensure_model_exists('panns_base')
            model_wrapper = AudioTagging(checkpoint_path=str(path_base), device='cpu')
            model = model_wrapper.model
            model.forward = types.MethodType(forward_panns_fix, model)
            model.fc_audit = nn.Linear(2048, num_classes)
            n_mels = 64
            checkpoint = torch.load(path_pesos, map_location=device)
            if 'model_state_dict' in checkpoint: checkpoint = checkpoint['model_state_dict']
            model.load_state_dict(checkpoint, strict=False)
            model.to(device)
            model.eval()
            return model, n_mels

        elif model_key == 'passt':
            try:
                full_model = get_basic_model(mode="logits")
                full_model.net.head = nn.Linear(768, num_classes)
            except NameError: return None, 128
            n_mels = 128
            checkpoint = torch.load(path_pesos, map_location=device)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            new_state_dict = {k.replace("net.", "") if k.startswith("net.") else k: v for k, v in state_dict.items()}
            try: full_model.net.load_state_dict(new_state_dict, strict=True)
            except: full_model.net.load_state_dict(new_state_dict, strict=False)
            full_model.net.to(device)
            full_model.net.eval()
            return full_model.net, n_mels
    except Exception as e:
        print(f"Error fatal: {e}")
        return None, 64

# --- PREDICCIÓN BASE ---
def predict_bird(audio_path, model_key):
    """Devuelve el vector crudo de probabilidades (NumPy array)"""
    model, n_mels_req = load_model(model_key)
    if model is None: return None

    tensor = process_audio(audio_path, n_mels=n_mels_req)
    if tensor is None: return None
    
    device = getattr(config, 'DEVICE', 'cpu')
    tensor = tensor.to(device)
    
    with torch.no_grad():
        output = model(tensor)
        if isinstance(output, dict): output = output.get('clipwise_output', output.get('logits'))
        if isinstance(output, tuple): output = output[0]        
        max_logit = torch.max(output)
        if max_logit > 5.0:
            output_ajustado = output - max_logit + 5.0
        else:
            output_ajustado = output
        probs = torch.softmax(output_ajustado,dim=1) 
        return probs.cpu().numpy()[0]

# --- NUEVA FUNCIÓN: DECODIFICADOR TOP 3 ---
def decode_top3(probs,threshold=0.20):
    """
    Métrica de calidad: Filtra por umbral de confianza en lugar de un número fijo.
    """
    sorted_index = np.argsort(probs)[::-1]
    
    results = []
    for idx in sorted_index:
        score = float(probs[idx])
        if score < threshold:
            break
        try:
            label = config.SPECIES_LIST[idx]
        except:
            label = f"Unknown_ID_{idx}"
            
        results.append({"label": label, "score": score})
    
    if not results and len(sorted_index) > 0:
        idx = sorted_index[0]
        results.append({"label": config.SPECIES_LIST[idx], "score": float(probs[idx])})
        
    return results