import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/detections.csv")

plt.figure(figsize=(8,4))
plt.plot(df["frame"], df["detections"], color="blue")
plt.xlabel("Frame")
plt.ylabel("Number of Detections")
plt.title("Detections per Frame (YOLOv8)")
plt.grid(True)
plt.savefig("results/detections_plot.png", dpi=300)
plt.show()

print("Gráfico guardado en results/detections_plot.png")
