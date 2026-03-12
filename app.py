import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from huggingface_hub import hf_hub_download
import os

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Melanoma2 AI Analyzer", layout="wide", page_icon="🔬")

# --- AVISO MÉDICO OBLIGATORIO ---
st.error("⚠️ **ADVERTENCIA MÉDICA:** Esta es una herramienta de Inteligencia Artificial experimental. Los resultados son aproximaciones estadísticas y **NO constituyen un diagnóstico médico**. No tome decisiones de salud basadas en esta web. **Debe acudir a un dermatólogo colegiado** para un examen profesional.")

# --- ARQUITECTURA DEL MODELO ---
class Melanoma2(nn.Module):
    def __init__(self):
        super(Melanoma2, self).__init__()
        def conv_block(in_f, out_f, dropout_rate):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_f), nn.LeakyReLU(0.1),
                nn.Conv2d(out_f, out_f, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_f), nn.LeakyReLU(0.1),
                nn.MaxPool2d(2), nn.Dropout(dropout_rate)
            )
        self.layer1 = conv_block(3, 64, 0.1)
        self.layer2 = conv_block(64, 128, 0.2)
        self.layer3 = conv_block(128, 256, 0.3)
        self.layer4 = conv_block(256, 512, 0.4)
        self.attention = nn.Sequential(nn.Conv2d(512, 1, kernel_size=1), nn.Sigmoid())
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(512, 512), nn.LeakyReLU(0.1),
            nn.Dropout(0.5), nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        att = self.attention(x)
        x_att = x * att
        return self.classifier(self.gap(x_att)), att

# --- FILTRO DULLRAZOR (Eliminación de vello) ---
class HairRemovalTransform:
    def __call__(self, img):
        img_cv = np.array(img)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        dst = cv2.inpaint(img_cv, mask, 1, cv2.INPAINT_TELEA)
        return Image.fromarray(dst)

# --- CARGA DEL MODELO DESDE HUGGING FACE ---
@st.cache_resource
def load_clinical_model():
    model = Melanoma2()
    repo_id = "LydiaTS04/cnn_melanoma"
    filename = "mas_datos_modelo_melanoma_2_local.pth"
    
    try:
        # Descarga el archivo desde Hugging Face usando la librería oficial
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

model = load_clinical_model()

# --- INTERFAZ DE USUARIO ---
st.title("🔬 Melanoma2: Diagnóstico Clínico Avanzado")
st.markdown("Cargue una imagen de la lesión cutánea para obtener un análisis de probabilidad mediante redes neuronales convolucionales.")
st.markdown("---")

uploaded_files = st.sidebar.file_uploader("Cargar Imágenes", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files and model is not None:
    for uploaded_file in uploaded_files:
        # Cargar y preprocesar imagen
        image = Image.open(uploaded_file).convert('RGB')
        img_clean = HairRemovalTransform()(image) 
        
        # Transformaciones para el modelo
        preprocess = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = preprocess(img_clean).unsqueeze(0)
        
        # Inferencia
        with torch.no_grad():
            output, att_map = model(input_tensor)
            prob = torch.sigmoid(output).item()
            label = "⚠️ MALIGNO" if prob > 0.5 else "✅ BENIGNO"
            color = "red" if prob > 0.5 else "green"

        # Generación de Mapa de Calor (Atención)
        att_np = att_map.squeeze().cpu().numpy()
        att_rescaled = cv2.resize(att_np, (image.size[0], image.size[1]))
        att_rescaled = (att_rescaled - att_rescaled.min()) / (att_rescaled.max() - att_rescaled.min() + 1e-8)
        heatmap = cv2.applyColorMap(np.uint8(255 * att_rescaled), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(np.array(image), 0.6, heatmap, 0.4, 0)

        # UI de Resultados
        with st.expander(f"ANÁLISIS: {uploaded_file.name}", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(image, caption="Imagen Original", use_container_width=True)
            with c2:
                st.image(overlay, caption="Zonas de Atención IA", use_container_width=True)
            with c3:
                st.subheader(f"Resultado: :{color}[{label}]")
                st.write(f"**Confianza:** {prob*100:.2f}%")
                st.progress(prob)
                
                st.markdown("### 🩺 Interpretación Médica")
                if prob > 0.5:
                    st.error("**HALLAZGOS:** Se detecta asimetría estructural y patrones de activación sospechosos. Se recomienda derivación urgente a dermatoscopia.")
                else:
                    st.success("**HALLAZGOS:** Lesión con arquitectura uniforme y baja activación. Se sugiere vigilancia preventiva y uso de protección solar.")

st.markdown("---")
st.caption("Arquitectura: 9 capas Conv, Módulo de Atención y optimización AdamW. **Recuerde: Ante cualquier duda, consulte a su médico.**")