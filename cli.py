import os
import cv2
import torch
import numpy as np
import argparse
from torchvision import transforms
from tqdm import tqdm
from ffmpeg_progress_yield import FfmpegProgress
from rvm import RobustVideoMatting  # Charger le modèle RVM

# Chemin du modèle
MODEL_PATH = "rvm_mobilenetv3.pth"

# Initialisation du modèle RVM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RobustVideoMatting()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device).eval()

def extract_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Extraction de {frame_count} frames...")
    
    for i in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{output_folder}/{i:05d}.png", frame)
    cap.release()

def remove_background(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor()])
    
    files = sorted(os.listdir(input_folder))
    
    for file in tqdm(files):
        img = cv2.imread(os.path.join(input_folder, file))
        img = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            _, _, matte = model(img)
        
        matte = matte.squeeze().cpu().numpy()
        matte = (matte * 255).astype(np.uint8)
        
        bg_removed = cv2.cvtColor(matte, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(os.path.join(output_folder, file), bg_removed)

def reconstruct_video(frame_folder, audio_file, output_video):
    print("Reconstruction de la vidéo...")
    cmd = [
        "ffmpeg", "-y", "-framerate", "30",
        "-i", f"{frame_folder}/%05d.png",
        "-i", audio_file, "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", output_video
    ]
    ff = FfmpegProgress(cmd)
    for progress in ff.run_command_with_progress():
        print(f"Progression : {progress}%")

def main():
    parser = argparse.ArgumentParser(description="Suppression du background d'une vidéo")
    parser.add_argument("video", type=str, help="Chemin de la vidéo")
    parser.add_argument("output", type=str, help="Vidéo de sortie")
    
    args = parser.parse_args()
    
    tmp_frames = "frames"
    tmp_processed = "processed"
    
    extract_frames(args.video, tmp_frames)
    remove_background(tmp_frames, tmp_processed)
    reconstruct_video(tmp_processed, args.video, args.output)

if __name__ == "__main__":
    main()
