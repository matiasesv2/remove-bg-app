import streamlit as st
import torch
import torchvision.transforms as T
import cv2
import numpy as np
import tempfile
from PIL import Image

# Configurar modelo DeepLabV3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
model.to(device).eval()

def segment_frame(frame):
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)['out'][0].cpu()
    mask = output.argmax(0).byte().numpy()
    mask = cv2.medianBlur(mask.astype(np.uint8) * 255, 5)
    return np.where(mask > 128, 255, 0).astype(np.uint8)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    output_path = tempfile.NamedTemporaryFile(suffix=".mov", delete=False).name
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'png '), fps, (width, height), True)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mask = segment_frame(frame_pil)
        b, g, r = cv2.split(frame)
        alpha = mask  # Canal alpha
        rgba_frame = cv2.merge((b, g, r, alpha))
        out.write(rgba_frame)
    
    cap.release()
    out.release()
    return output_path

st.title("🎥 Eliminar Fondo de Videos con IA")

uploaded_file = st.file_uploader("Sube un video (MP4, MOV)", type=["mp4", "mov"])

if uploaded_file is not None:
    temp_video = tempfile.NamedTemporaryFile(delete=False)
    temp_video.write(uploaded_file.read())
    
    st.video(temp_video.name)
    
    if st.button("Procesar Video"):
        with st.spinner("Procesando..."):
            output_video = process_video(temp_video.name)
        st.success("Procesamiento completado!")
        st.download_button("Descargar Video", open(output_video, "rb"), "video_sin_fondo.mov", "video/quicktime")
