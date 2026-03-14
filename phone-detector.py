import cv2
from ultralytics import YOLO

# 1. Load a standard Detection model (using YOLO11 for stability)
model = YOLO('yolo11n.pt') 

cap = cv2.VideoCapture(0)

print("Starting Phone Detection... Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 2. Run inference
    # classes=[67] tells YOLO to only care about cell phones
    results = model(frame, stream=True, classes=[67], conf=0.5)

    for r in results:
        # Check if any phones were found in the frame
        if len(r.boxes) > 0:
            for box in r.boxes:
                # Get coordinates for the box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # Draw the box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, f"PHONE DETECTED: {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Print to console for logic/triggers
                print("⚠️ Cell phone detected in frame!")

    cv2.imshow("Phone Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()