import cv2
from inference_sdk import InferenceHTTPClient

# Inicializa el cliente
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="ZGLuDrCQpJdtcfEuYUFL"
)

# Ruta al vídeo original
video_path = "../../videos/Verstappen Wins Title With Final Lap Overtake!  2021 Abu Dhabi Grand Prix - FORMULA 1 (1080p, h264).mp4" 
cap = cv2.VideoCapture(video_path)

# Parámetros del vídeo de salida
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("output_f1_detected.mp4", fourcc, fps, 
(width, height))

frame_count = 0
max_frames = 200

while frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break

    # Guardamos el frame temporalmente como imagen
    temp_path = "temp.jpg"
    cv2.imwrite(temp_path, frame)

    # Inferencia con Roboflow
    try:
        result = CLIENT.infer(temp_path, 
model_id="detection-f1-cars/11")
    except Exception as e:
        print(f"Error en inferencia del frame {frame_count}: {e}")
        continue

    # Dibujar predicciones
    for pred in result["predictions"]:
        x = int(pred["x"] - pred["width"] / 2)
        y = int(pred["y"] - pred["height"] / 2)
        w = int(pred["width"])
        h = int(pred["height"])
        label = pred["class"]
        conf = pred["confidence"]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 
0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x, y - 
10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 
0), 2)

    out.write(frame)
    frame_count += 1
    print(f"Procesado frame {frame_count}")

cap.release()
out.release()
cv2.destroyAllWindows()
print(" Vídeo guardado como output_f1_detected.mp4")

