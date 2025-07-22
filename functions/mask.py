import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

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


def red_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 100])
    upper_red2 = np.array([179, 255, 255])


    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.title("Original")
    
    plt.subplot(1,2,2)
    plt.imshow(mask, cmap = "gray")
    plt.title("Masked")

    plt.show()

    return None



def green_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 50, 100])    # H, S, V
    upper_green = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.title("Original")
    
    plt.subplot(1,2,2)
    plt.imshow(mask, cmap = "gray")
    plt.title("Green_Masked")

    plt.show()

    return None



def brown_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_brown = np.array([0, 30, 20])     # H, S, V
    upper_brown = np.array([30, 120, 150])

    mask = cv2.inRange(hsv, lower_brown, upper_brown)
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.title("Original")
    
    plt.subplot(1,2,2)
    plt.imshow(mask, cmap = "gray")
    plt.title("Brown_Masked")

    plt.show()

    return None
    

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
    blue_bg = np.full_like(image, (255, 0, 0))  # BGR = Blue

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


def hsv_slider(image):
    def nothing(x):
        pass
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create window
    cv2.namedWindow('HSV Adjuster')

    # Create trackbars for H, S, V
    cv2.createTrackbar('H1 Lower','HSV Adjuster',0,179,nothing)
    cv2.createTrackbar('H1 Upper','HSV Adjuster',179,179,nothing)
    cv2.createTrackbar('H2 Lower','HSV Adjuster',0,179,nothing)
    cv2.createTrackbar('H2 Upper','HSV Adjuster',179,179,nothing)
    cv2.createTrackbar('S Lower','HSV Adjuster',0,255,nothing)
    cv2.createTrackbar('S Upper','HSV Adjuster',255,255,nothing)
    cv2.createTrackbar('V Lower','HSV Adjuster',0,255,nothing)
    cv2.createTrackbar('V Upper','HSV Adjuster',255,255,nothing)


    while True:
        # Get current positions
        hL1 = cv2.getTrackbarPos('H1 Lower','HSV Adjuster')
        hU1 = cv2.getTrackbarPos('H1 Upper','HSV Adjuster')
        hL2 = cv2.getTrackbarPos('H2 Lower','HSV Adjuster')
        hU2 = cv2.getTrackbarPos('H2 Upper','HSV Adjuster')
        sL = cv2.getTrackbarPos('S Lower','HSV Adjuster')
        sU = cv2.getTrackbarPos('S Upper','HSV Adjuster')
        vL = cv2.getTrackbarPos('V Lower','HSV Adjuster')
        vU = cv2.getTrackbarPos('V Upper','HSV Adjuster')

        # Create lower and upper bounds
        lower1 = np.array([hL1, sL, vL])
        upper1 = np.array([hU1, sU, vU])
        lower2 = np.array([hL2, sL, vL])
        upper2 = np.array([hU2, sU, vU])
        # Create mask and result image
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        result = cv2.bitwise_and(image, image, mask=mask)

        # Show results
        cv2.imshow('Original', image)
        cv2.imshow('Mask', mask)
        cv2.imshow('Filtered', result)

        # Break with ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


def canny(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 1.4)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, threshold1=100, threshold2=200)

    # Show original and edge-detected image
    cv2.imshow('Original', image)
    cv2.imshow('Canny Edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test(image):
    """
    img = image
    colors = ('b', 'g', 'r')

    plt.figure(figsize=(10,5))
    for i, color in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])

    plt.title('BGR Channel Histograms')
    plt.xlabel('Intensity')
    plt.ylabel('Pixel Count')
    plt.show()
    """

    import cv2
    import numpy as np
    from sklearn.decomposition import PCA

    img = image
    img_flat = img.reshape((-1, 3))

    # Perform PCA reduction to 2 components
    pca = PCA(n_components=2)
    img_pca = pca.fit_transform(img_flat)

    # Reshape back to image format
    img_2ch = img_pca.reshape((img.shape[0], img.shape[1], 2))

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img_2ch[:, :, 0], cmap='gray')
    ax[0].set_title('Channel 1')

    ax[1].imshow(img_2ch[:, :, 1], cmap='gray')
    ax[1].set_title('Channel 2')

    plt.show()


def image_transform(img):
    # --- Load and preprocess image ---
    img_float = img.astype(np.float32) / 255.0
    B, G, R = cv2.split(img_float)

    # --- Channel 1: Red - Green (color emphasis) ---
    color_axis = R - G
    color_axis_norm = cv2.normalize(color_axis, None, 0, 255, cv2.NORM_MINMAX)
    color_axis_norm = color_axis_norm.astype(np.uint8)

    # --- Channel 2: Brightness ---
    brightness = 0.299 * R + 0.587 * G + 0.114 * B
    brightness_norm = (brightness * 255).clip(0, 255).astype(np.uint8)

    # --- Channel 3: Canny edge detection ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)  # Adjust thresholds as needed

    # --- Combine into 3-channel output ---
    output_3ch = cv2.merge([color_axis_norm, brightness_norm, edges])

    cv2.imshow("3ch", output_3ch)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return output_3ch

for f in train_filenames:
    image, data = load_file(train_path, f)

    # hsv_slider(image)
    # key_mask(image)
    # canny(image)
    
    image_transform(key_mask(image))