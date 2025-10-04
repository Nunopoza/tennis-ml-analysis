import cv2

video_path = "results/tennis_detected.mp4"
output_path = "results/tennis_tracked.mp4"

cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Tracker (CSRT funciona bien)
tracker = cv2.TrackerCSRT_create()

initBB = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if initBB is not None:
        success, box = tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Tracking", frame)
    out.write(frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        initBB = cv2.selectROI("Tracking", frame, fromCenter=False, showCrosshair=True)
        tracker.init(frame, initBB)
    elif key == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
