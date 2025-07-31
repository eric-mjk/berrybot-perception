import cv2
import torch
from ultralytics import YOLO
import numpy as np
from funcs import check_data, make_filenames, load_file, show_image, prediction_handling, print_data

# Load your trained YOLOv8 model
model_path = 'Models/model1.pt'
model = YOLO(model_path)

# Initialize webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Run YOLOv8 prediction on the frame
    results = model.predict(source=frame, conf=0.4, save=False, verbose=False)

    handled_results = prediction_handling(results)
    print(handled_results)

    for r in results:
        boxes = r.boxes
        keypoints = r.keypoints

        # Draw bounding boxes
        if boxes is not None:
            for box in boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw keypoints and lines
        if keypoints is not None:
            for kp in keypoints.xy:
                points = []
                for x, y in kp:
                    if x > 0 and y > 0:
                        pt = (int(x), int(y))
                        points.append(pt)
                        cv2.circle(frame, pt, 3, (0, 0, 255), -1)
                    else:
                        points.append(None)

    # Show the frame
    cv2.imshow("YOLOv8 Real-Time Prediction", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
