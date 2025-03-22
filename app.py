import streamlit as st
import torch
import numpy as np
import cv2
import tempfile
import os
import imageio
from moviepy.editor import VideoFileClip

# Télécharger le modèle Robust Video Matting (RVM) de PyTorch
@st.cache_resource
def load_model():
    model = torch.hub.load("pytorch/vision:v0.10.0", "deeplabv3_resnet101", pretrained=True)
    model.eval()
    return model

model = load_model()

# Fonction pour extraire et traiter les frames d’une vidéo
def process_video(input_path, output_path):
    vid = imageio.get_reader(input_path, "ffmpeg")
    fps = vid.get_meta_data()["fps"]
    output_frames = []

    for frame in vid:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convertir en BGR pour OpenCV
        frame_tensor = torch.tensor(frame).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        
        with torch.no_grad():
            output = model(frame_tensor)["out"]
            mask = output.argmax(1).squeeze().cpu().numpy()
        
        mask = (mask * 255).astype(np.uint8)
        frame_no_bg = cv2.bitwise_and(frame, frame, mask=mask)
        output_frames.append(frame_no_bg)

    # Sauvegarder la vidéo traitée
    imageio.mimsave(output_path, output_frames, fps=fps)

# Interface Streamlit
st.title("🔍 Suppression de Background Vidéo")
uploaded_file = st.file_uploader("Chargez une vidéo", type=["mp4", "mov", "avi"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    st.text("⏳ Traitement en cours...")
    process_video(tfile.name, output_file)
    
    st.video(output_file)
    st.download_button("📥 Télécharger la vidéo", output_file, file_name="video_sans_bg.mp4")
