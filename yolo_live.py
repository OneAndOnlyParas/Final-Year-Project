import cv2
import time
from ultralytics import YOLO
import winsound  # for beep sound (Windows)

# ---------------- LOAD MODEL ----------------
model = YOLO("yolov8n.pt")  # lightweight model

# ---------------- WEBCAM ----------------
cap = cv2.VideoCapture(0)

# ---------------- ALERT CONTROL ----------------
last_alert_time = 0
ALERT_COOLDOWN = 5  # seconds

print("Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---------------- YOLO DETECTION ----------------
    results = model(frame, conf=0.5)

    detected_animals = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            # ---------------- FILTER ONLY ANIMALS ----------------
            animal_classes = [
                "dog", "cat", "cow", "sheep", "horse", "elephant",
                "bear", "zebra", "giraffe", "deer"
            ]

            if label in animal_classes:
                detected_animals.append(label)

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw bounding box
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # ---------------- ALERT ----------------
    current_time = time.time()

    if detected_animals and (current_time - last_alert_time > ALERT_COOLDOWN):
        print(f"ALERT! Animal detected: {set(detected_animals)}")

        # Beep sound (Windows)
        winsound.Beep(1000, 500)

        last_alert_time = current_time

    # ---------------- DISPLAY ----------------
    cv2.imshow("YOLO Animal Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()