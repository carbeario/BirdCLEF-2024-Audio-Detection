import torch
import torch.nn as nn
import torch.nn.functional as F
import types
import streamlit as st
import os
import numpy as np
from huggingface_hub import hf_hub_download

# Intentamos importar PaSST
try:
    from hear21passt.base import get_basic_model
except ImportError:
    pass

from panns_inference import AudioTagging
from src import config
from src.audio_utils import process_audio

# --- CONFIGURACIÓN DEL REPOSITORIO ---
HF_REPO_ID = "CarLSBR/BirdCLEF-2024-PANNs" 

# ==========================================
# 1. PARCHE PANNs
# ==========================================
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

# ==========================================
# 2. GESTOR DE DESCARGAS
# ==========================================
def ensure_model_exists(model_key):
    local_path = config.MODEL_PATHS[model_key]
    if os.path.exists(local_path):
        return local_path
    
    filename = os.path.basename(local_path)
    folder = os.path.dirname(local_path)
    
    print(f"⬇️ [SYSTEM] Descargando {filename} desde Hugging Face...")
    try:
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            local_dir=folder,
            local_dir_use_symlinks=False
        )
        print("✅ Descarga completada.")
        return local_path
    except Exception as e:
        print(f"❌ Error descargando modelo: {e}")
        return None

# ==========================================
# 3. CARGA DE MODELOS (CORREGIDA Y DEBUGGEADA)
# ==========================================

@st.cache_resource
def load_model(model_key):
    device = config.DEVICE
    num_classes = len(config.SPECIES_LIST)
    
    # 1. Buscamos el path del modelo FINO (El tuyo)
    path_pesos = ensure_model_exists(model_key)
    if not path_pesos: return None, 64

    print(f"--> Cargando arquitectura: {model_key.upper()}")

    try:
        if model_key == 'panns':
            # --- CARGA PANNS ROBUSTA ---
            
            # A. Asegurar que existe el ESQUELETO BASE (Cnn14...) en weights/
            path_base = ensure_model_exists('panns_base')
            
            # B. Inicializar librería apuntando al archivo base explícito
            # Al pasar checkpoint_path, evitamos que cree carpetas basura
            model_wrapper = AudioTagging(checkpoint_path=str(path_base), device='cpu')
            
            model = model_wrapper.model
            model.forward = types.MethodType(forward_panns_fix, model)
            
            # C. Ajustar la última capa a 101 clases (Cirugía)
            model.fc_audit = nn.Linear(2048, num_classes)
            n_mels = 64
            
            # D. Cargar TUS pesos entrenados (El Cerebro)
            checkpoint = torch.load(path_pesos, map_location=device)
            if 'model_state_dict' in checkpoint: checkpoint = checkpoint['model_state_dict']
            
            # strict=False es vital porque la base tenía 527 clases y tú cargas 101
            model.load_state_dict(checkpoint, strict=False)
            
            model.to(device)
            model.eval()
            return model, n_mels

        elif model_key == 'passt':
            # --- CARGA PASST (Igual que tenías, funciona bien) ---
            try:
                full_model = get_basic_model(mode="logits")
                full_model.net.head = nn.Linear(768, num_classes)
            except NameError:
                return None, 128
            n_mels = 128
        
            checkpoint = torch.load(path_pesos, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("net."):
                    new_key = k.replace("net.", "")
                    new_state_dict[new_key] = v
                else:
                    new_state_dict[k] = v
            
            try:
                full_model.net.load_state_dict(new_state_dict, strict=True)
            except RuntimeError:
                full_model.net.load_state_dict(new_state_dict, strict=False)
            
            full_model.net.to(device)
            full_model.net.eval()
            return full_model.net, n_mels

    except Exception as e:
        print(f"❌ Error fatal cargando {model_key}: {e}")
        return None, 64
    
# ==========================================
# 4. PREDICCIÓN
# ==========================================
def predict_bird(audio_path, model_key):
    model, n_mels_req = load_model(model_key)
    
    if model is None: return None

    tensor = process_audio(audio_path, n_mels=n_mels_req)
    if tensor is None: return None
    
    tensor = tensor.to(config.DEVICE)
    
    with torch.no_grad():
        output = model(tensor)
        
        if isinstance(output, dict): 
            output = output.get('clipwise_output', output.get('logits'))
        if isinstance(output, tuple): 
            output = output[0]
            
        probs = F.softmax(output, dim=1)
        return probs.cpu().numpy()[0]