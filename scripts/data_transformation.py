import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

# --- Configuration ---
input_folder = "Strawberry Feature Points.v11-complete_dataset.yolov8\\valid\\images"
output_folder = "masked_dataset.yolov8\\valid\\images"
os.makedirs(output_folder, exist_ok=True)

# --- Your predefined image processing function ---
def key_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 0])
    upper_red1 = np.array([30, 255, 255])
    lower_red2 = np.array([160, 50, 0])
    upper_red2 = np.array([179, 255, 255])


    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask = cv2.bitwise_or(mask1, mask2)

    # 1. Invert the mask
    inv_mask = cv2.bitwise_not(mask)

    # 2. Create a background (e.g., blue)
    blue_bg = np.full_like(image, (0, 0, 0))  # BGR = Blue

    # 3. Extract masked regions from original
    foreground = cv2.bitwise_and(image, image, mask=mask)

    # 4. Extract background regions from the blue image
    background = cv2.bitwise_and(blue_bg, blue_bg, mask=inv_mask)

    # 5. Combine the two
    result = cv2.add(foreground, background)

    # result = cv2.bitwise_and(image, image, mask=mask)

    # while True:   
    #     cv2.imshow("Original", image) 
    #     cv2.imshow("Result", result)
    #     # Check for exit
    #     if cv2.waitKey(1) & 0xFF == 27:  # ESC key
    #         break


    return result


print(os.listdir(input_folder))

# --- Process all .png files ---
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".jpg"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Load, process, and save image
        img = cv2.imread(input_path)
        processed = key_mask(img)
        cv2.imwrite(output_path, processed)

        print(f"Processed and saved: {output_path}")
