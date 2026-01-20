import torch
import torch.nn.functional as F
import types
import streamlit as st
from panns_inference import AudioTagging
from src import config, audio_utils
from huggingface_hub import hf_hub_download
import os






# ==========================================
# 1. PARCHE DE ARQUITECTURA (CRÍTICO)
# ==========================================
def forward_panns_fix(self, x, mixup_lambda=None):
    x = x[:, 0:1, :, :]   # (Batch, 1, Mel, Time)
    x = x.transpose(2, 3) # (Batch, 1, Time, Mel)
    x = x.transpose(1, 3) # (Batch, 64, Time, 1)
    x = self.bn0(x)       
    x = x.transpose(1, 3) # (Batch, 1, Time, Mel)
    
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
# 2. CARGA DEL MODELO
# ==========================================
@st.cache_resource
def cargar_modelo():
    device_actual = config.DEVICE 
    print(f"--> Cargando modelo PANNs desde: {config.MODEL_PATH}")
    
    try:
        model_wrapper = AudioTagging(checkpoint_path=None, device=device_actual)
        model = model_wrapper.model
        
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        
        model.forward = types.MethodType(forward_panns_fix, model)
        model.fc_audit = torch.nn.Linear(2048, len(config.SPECIES_LIST))
        
        state_dict = torch.load(config.MODEL_PATH, map_location=torch.device(device_actual))
        model.load_state_dict(state_dict)
        model.to(device_actual)
        model.eval()
        
        return model
        
    except Exception as e:
        print(f"❌ Error fatal: {e}")
        return None

# ==========================================
# 3. FUNCIÓN DE PREDICCIÓN (CON MEDIA PONDERADA)
# ==========================================
def predecir(model, audio_path):
    """
    Estrategia 'TOP-K CONSENSUS':
    1. Analiza todas las ventanas del audio.
    2. Selecciona solo las 3 ventanas con mayor 'seguridad' (confianza).
    3. Promedia esas 3 para dar el veredicto final.
    
    Esto elimina totalmente el efecto del silencio o el ruido de fondo.
    """
    # 1. Obtener batch de fragmentos
    tensor_batch = audio_utils.procesar_audio(audio_path)
    
    if tensor_batch is None or len(tensor_batch) == 0:
        return None
    
    tensor_batch = tensor_batch.to(config.DEVICE)
    
    with torch.no_grad():
        output = model(tensor_batch)
        if isinstance(output, dict): output = output['clipwise_output']
        if isinstance(output, tuple): output = output[0]
        
        # Probabilidades por ventana (N_ventanas x N_clases)
        probs_per_chunk = F.softmax(output, dim=1)
        
        # --- LÓGICA TOP-K (La clave para subir el %) ---
        
        # 1. ¿Qué tan seguro está el modelo en cada ventana?
        # Obtenemos la probabilidad más alta de cada fila (la "confianza")
        window_confidences, _ = torch.max(probs_per_chunk, dim=1) # Shape: [N]
        
        # 2. Elegimos las mejores ventanas (ej: Top 3 o Top 5)
        # Si el audio es muy corto y hay menos de 3, cogemos las que haya
        k = min(3, len(window_confidences)) 
        
        # Obtenemos los índices de las ventanas con mayor confianza
        top_k_values, top_k_indices = torch.topk(window_confidences, k)
        
        # 3. Filtramos: Nos quedamos solo con las predicciones de esas ventanas "buenas"
        best_windows_probs = probs_per_chunk[top_k_indices] # Shape: [3, 101]
        
        # 4. Promediamos solo las mejores
        final_probs = torch.mean(best_windows_probs, dim=0)
        
    return final_probs.cpu().numpy()


@st.cache_resource
def cargar_modelo():
    device_actual = config.DEVICE 
    
    # --- NUEVA LÓGICA DE HUGGING FACE (Corregida) ---
    repo_id = "CarLSBR/BirdCLEF-2024-PANNs"
    filename = "panns_inference.pth" 
    
    # Aseguramos que la carpeta de destino exista
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)

    if not os.path.exists(config.MODEL_PATH):
        # Usamos st.empty() para que el mensaje desaparezca después de la descarga
        status_placeholder = st.empty()
        status_placeholder.info("Primer inicio: Descargando pesos del modelo (328MB)...")
        try:
            hf_hub_download(
                repo_id=repo_id, 
                filename=filename, 
                local_dir=os.path.dirname(config.MODEL_PATH),
                local_dir_use_symlinks=False
            )
            
            path_descargado = os.path.join(os.path.dirname(config.MODEL_PATH), filename)
            if path_descargado != config.MODEL_PATH:
                # Si en config.py el nombre es distinto, lo renombramos
                os.rename(path_descargado, config.MODEL_PATH)
            
            status_placeholder.success("Modelo listo.")
        except Exception as e:
            st.error(f"Error al descargar desde Hugging Face: {e}")
            return None
    # -------------------------------------

    print(f"--> Cargando modelo PANNs desde: {config.MODEL_PATH}")
    
    try:
        # Aquí sigue tu lógica de AudioTagging, parcheo y load_state_dict...
        model_wrapper = AudioTagging(checkpoint_path=None, device=device_actual)
        model = model_wrapper.model
        
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        
        model.forward = types.MethodType(forward_panns_fix, model)
        model.fc_audit = torch.nn.Linear(2048, len(config.SPECIES_LIST))
        
        state_dict = torch.load(config.MODEL_PATH, map_location=torch.device(device_actual))
        model.load_state_dict(state_dict)
        model.to(device_actual)
        model.eval()
        
        return model
    except Exception as e:
        st.error(f"❌ Error fatal al cargar el modelo: {e}")
        return None