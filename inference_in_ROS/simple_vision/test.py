import cv2
import numpy as np

# Initialize webcam (0 is usually the default camera)
cap = cv2.VideoCapture(4)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break


    img = cv2.resize(frame, (640, 640))
    blurred = cv2.GaussianBlur(img, (7,7), 0)

    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # Red wraps around the hue range; define two ranges
    lower_red1 = np.array([0,  150,  50])
    upper_red1 = np.array([30, 255, 255])
    lower_red2 = np.array([170, 100,  50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask  = cv2.bitwise_or(mask1, mask2)
    
    # cv2.imshow('Strawberry real', img)
    # cv2.imshow('Strawberry mask', mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:           # filter out tiny blobs
            continue
        # optional: filter by shape (solidity, aspect ratio) to reject non-berry shapes
        x,y,w,h = cv2.boundingRect(cnt)
        aspect = w/float(h)
        if aspect < 0.1 or aspect > 10.0:
            continue
        boxes.append((x,y,w,h))

    for (x,y,w,h) in boxes:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.imshow('Strawberry Detections', img)
    # cv2.waitKey(0)
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



"""

template = cv2.imread('strawberry_template.jpg',0)
res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
loc = np.where(res >= 0.6)
# draw boxes around matches

"""