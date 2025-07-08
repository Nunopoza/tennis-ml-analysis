import cv2
import os

# Ruta al vídeo (ajusta si cambiaste el nombre o carpeta)
video_path = "data/raw/Alcaraz_recorte2.mp4"
output_dir = "data/frames/frames2"
os.makedirs(output_dir, exist_ok=True)

# Leer vídeo
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps * 0.5)  # Extrae 1 frame cada 0.5 segundos

frame_count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        filename = f"{output_dir}/frame_{saved_count:04d}.jpg"
        cv2.imwrite(filename, frame)
        saved_count += 1

    frame_count += 1

cap.release()
print(f" {saved_count} frames guardados en {output_dir}")

