import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from mpl_toolkits.mplot3d import Axes3D


from funcs import check_data, make_filenames, load_file, show_image

# Target Directory
path = "TEST2/Strawberry Feature Points.v8i.yolov8"

train_path = os.path.join(path, "train")
test_path = os.path.join(path, "test")
valid_path = os.path.join(path, "valid")

# List all files - this exlcudes the extension ()
train_filenames = make_filenames(train_path)
test_filenames = make_filenames(test_path)
valid_filenames = make_filenames(valid_path)



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

    while True:   
        cv2.imshow("Original", image) 
        cv2.imshow("Result", result)
        # Check for exit
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break


    return result


def image_transform(img, normalize=True):
    img_float = img.astype(np.float32) / 255.0
    B, G, R = cv2.split(img_float)

    color_axis = R - G
    brightness = 0.299 * R + 0.587 * G + 0.114 * B
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)

    if normalize:
        color_axis = cv2.normalize(color_axis, None, 0, 255, cv2.NORM_MINMAX)
        brightness = (brightness * 255).clip(0, 255)
    
    out = cv2.merge([
        color_axis.astype(np.uint8),
        brightness.astype(np.uint8),
        edges
    ])
    return out

def visualize_3d(img):

    # Load and convert image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize for clarity
    pixels = img_rgb.reshape(-1, 3)
    num_samples = int(0.01 * len(pixels))
    indices = np.random.choice(len(pixels), num_samples, replace=False)
    sampled_pixels = pixels[indices]

    r, g, b = sampled_pixels[:, 0], sampled_pixels[:, 1], sampled_pixels[:, 2]

    # Plot image and RGB 3D scatter
    fig = plt.figure(figsize=(14, 6))

    # Plot the image
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img_rgb)
    ax1.set_title("Image")
    ax1.axis("off")

    # Plot RGB color space
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(r, g, b, c=sampled_pixels / 255.0, marker='o', s=5)
    ax2.set_xlabel("Red")
    ax2.set_ylabel("Green")
    ax2.set_zlabel("Blue")
    ax2.set_title("RGB Pixel Distribution")

    plt.tight_layout()
    plt.show()

def visualize_3d_hsv(img):
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Convert BGR → HSV and also keep original RGB for coloring
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Flatten
    hsv_pixels = img_hsv.reshape(-1, 3)
    rgb_pixels = img_rgb.reshape(-1, 3)

    # Sample 1%
    num_samples = int(0.01 * len(hsv_pixels))
    indices = np.random.choice(len(hsv_pixels), num_samples, replace=False)
    sampled_hsv = hsv_pixels[indices]
    sampled_rgb = rgb_pixels[indices]

    h, s, v = sampled_hsv[:, 0], sampled_hsv[:, 1], sampled_hsv[:, 2]

    # Plot image and HSV 3D scatter
    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img_rgb)
    ax1.set_title("Image")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(h, s, v, c=sampled_rgb / 255.0, marker='o', s=5)
    ax2.set_xlabel("Hue")
    ax2.set_ylabel("Saturation")
    ax2.set_zlabel("Value")
    ax2.set_title("HSV Pixel Distribution")

    plt.tight_layout()
    plt.show()


for f in train_filenames:
    image, data = load_file(train_path, f)


    # _ = key_mask(image)
    # cv2.imshow("f", key_mask(image))
    # cv2.waitKey(0)
    # image_transform(key_mask(image))


    cv2.imshow("img",image_transform(image))
    cv2.waitKey(0)
