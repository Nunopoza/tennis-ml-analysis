from inference_sdk import InferenceHTTPClient
import os
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURACIÓN ---
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY") 
)

MODEL_ID = "tennis-objects-rw5zf/2"

input_dir = "data/frames"
output_dir = "results/frames_detected"
os.makedirs(output_dir, exist_ok=True)

video_output = "results/tennis_detected.mp4"
csv_output = "results/detections.csv"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = 30
out = None

import csv
with open(csv_output, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "label", "confidence"])

    for i, filename in enumerate(sorted(os.listdir(input_dir))):
        if not filename.endswith(".jpg"):
            continue
        frame_path = os.path.join(input_dir, filename)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        try:
            result = CLIENT.infer(frame_path, model_id=MODEL_ID)
        except Exception as e:
            print(f" Error al inferir {filename}: {e}")
            continue

        # Dibujar detecciones
        for pred in result.get("predictions", []):
            x = int(pred["x"] - pred["width"] / 2)
            y = int(pred["y"] - pred["height"] / 2)
            w = int(pred["width"])
            h = int(pred["height"])
            label = pred["class"]
            conf = pred["confidence"]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            writer.writerow([i, label, f"{conf:.2f}"])

        if out is None:
            height, width = frame.shape[:2]
            out = cv2.VideoWriter(video_output, fourcc, fps, (width, height))

        out.write(frame)
        cv2.imwrite(os.path.join(output_dir, filename), frame)
        print(f" Procesado {filename}")

if out:
    out.release()
print(f"\n Vídeo generado: {video_output}")
print(f" Detecciones guardadas en: {csv_output}")
