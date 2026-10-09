import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os

# ==========================================
# 1. CONFIGURACIÓN Y ESTILO VJ
# ==========================================
st.set_page_config(
    page_title="Gesture-Synth VJ",
    page_icon="🖐️",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syncopate:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Syncopate', sans-serif;
        background-color: #050505 !important;
        color: #FFFFFF !important;
    }
    
    .status-panel {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        transition: all 0.3s ease;
        border: 2px solid #333;
    }
    
    .active-trigger {
        border-color: #00FFCC;
        box-shadow: 0 0 20px rgba(0, 255, 204, 0.4);
        background-color: rgba(0, 255, 204, 0.1);
    }
    
    h1, h2, h3 { color: #00FFCC !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LÓGICA DE CARGA DEL MODELO
# ==========================================
@st.cache_resource
def cargar_modelo():
    try:
        return load_model('keras_model.h5', compile=False)
    except Exception:
        return None

# ==========================================
# 3. INTERFAZ PRINCIPAL
# ==========================================
st.title("🖐️ Gesture-Synth VJ Controller")
st.markdown("Entrena un modelo en **Teachable Machine**, súbelo como `keras_model.h5` y usa la cámara para activar escenas y moduladores de video mediante gestos u objetos físicos.")

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3565/3565893.png", width=80)
    st.subheader("Panel de Control")
    st.write("Conecta este clasificador por OSC a tu software de mezcla en tiempo real.")
    st.divider()
    st.write("⚙️ **Mapeo de Clases (Teachable Machine):**")
    st.write("• **Clase 0:** Estiven (Admin) / Reposo")
    st.write("• **Clase 1:** Gesto 1 -> Efecto Glitch")
    st.write("• **Clase 2:** Gesto 2 -> Emisor de Partículas")

col1, col2 = st.columns([1.5, 1])

with col1:
    img_file_buffer = st.camera_input("🎥 Sensor Óptico (Cámara)", key="cam_sensor")

with col2:
    st.write("### 🎛️ Estado del Sintetizador")
    
    # Contenedores dinámicos para los efectos visuales
    panel_placeholder = st.empty()
    
    if img_file_buffer is None:
        panel_placeholder.markdown("""
        <div class="status-panel">
            <h3 style="color: #666 !important;">Esperando input de video...</h3>
            <p>Enciende la cámara para comenzar el análisis.</p>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# 4. INFERENCIA Y PROCESAMIENTO
# ==========================================
if img_file_buffer is not None:
    model = cargar_modelo()
    
    if model is not None:
        # Preprocesamiento de la imagen según los estándares de Teachable Machine
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        img = Image.open(img_file_buffer).convert('RGB') # Asegurar RGB
        img = img.resize((224, 224))
        img_array = np.array(img)
        
        # Normalización (-1 a 1)
        normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array

        # Predicción
        with st.spinner("Analizando fotograma..."):
            prediction = model.predict(data)
            
            # Buscamos la clase con mayor probabilidad
            clase_detectada = np.argmax(prediction[0])
            probabilidad = prediction[0][clase_detectada]
            
            # Solo accionamos si la seguridad es mayor al 60%
            if probabilidad > 0.60:
                if clase_detectada == 0:
                    panel_placeholder.markdown(f"""
                    <div class="status-panel" style="border-color: #444;">
                        <h2 style="color: #FFF !important;">👤 Estiven (Admin)</h2>
                        <p>Estado: <b>Standby / Reposo</b></p>
                        <p style="font-size: 0.8em; color: #888;">Probabilidad: {probabilidad*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                elif clase_detectada == 1:
                    panel_placeholder.markdown(f"""
                    <div class="status-panel active-trigger" style="border-color: #FF003C; box-shadow: 0 0 20px rgba(255, 0, 60, 0.4); background-color: rgba(255, 0, 60, 0.1);">
                        <h2 style="color: #FF003C !important;">💥 TRIGGER 1 ACTIVO</h2>
                        <p>Activando: <b>Data Glitch / Aberración Cromática</b></p>
                        <p style="font-size: 0.8em; color: #888;">Probabilidad: {probabilidad*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                elif clase_detectada >= 2:
                    panel_placeholder.markdown(f"""
                    <div class="status-panel active-trigger" style="border-color: #BD00FF; box-shadow: 0 0 20px rgba(189, 0, 255, 0.4); background-color: rgba(189, 0, 255, 0.1);">
                        <h2 style="color: #BD00FF !important;">✨ TRIGGER 2 ACTIVO</h2>
                        <p>Activando: <b>Sistema de Partículas / Luma Key</b></p>
                        <p style="font-size: 0.8em; color: #888;">Probabilidad: {probabilidad*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                panel_placeholder.warning("Señal demasiado débil. Acerca el objeto a la cámara.")
                
    else:
        st.error("⚠️ No se encontró el archivo `keras_model.h5`. Asegúrate de subir el modelo descargado de Teachable Machine a tu repositorio.")
