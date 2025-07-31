import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path


"""
Prelimanary File Sturcture

- yolostylefolder
    - train
        -
        -
    - test
        -
        -
    - valid
        - images (.png)
        - labels (.txt)

"""



def make_filenames(dir):
    """
    make_filenames(str : dir) -> list[str]
        * dir : path of train/valid/test
        * list[str] : list of the filenames in the train/valid/test
        - returns the list of filenames (excluding the extensions)
    """
    img_path = os.path.join(dir, "images")
    label_path = os.path.join(dir, "labels")
    ret = os.listdir(img_path)
    
    filenames = []

    for f in ret:
        full_path = os.path.join(img_path, f)
        assert os.path.isfile(full_path), f"Not a file: {full_path}"
    
        base_name = os.path.splitext(f)[0]  # remove extension
        extent_name = base_name + ".txt"
        full_path = os.path.join(label_path, extent_name)
        assert os.path.isfile(full_path), f"Not exist in label: {full_path}"

        filenames.append(base_name)

    assert len(ret) == len(os.listdir(label_path)), f"image and label files have different length | size : {len(ret)}, {len(label_path)} | path : {dir}"
    
    return filenames



def check_data():
    """
    check_data()
        - check it the data is correct
        - the full body of make_filenames
    """
    _ = make_filenames



def load_file(main_path, filename):
    """
    load_file(str : main_path, str : filename) -> tuple(image, data)
        * main_path : path of train/valid/test
        * filename :  str of a filename (without extensions) inside the train/valid/test
        * tuple : (image, data)
            - image : cv2.image of file
            - data : list of  the following dictionary structure   -   [ {} , {} , {} ]
                data = {
                    "class" : 0,
                    "bbox"  : { "x_center" : double,
                            "y_center" : double,
                            "width"    : double,
                            "height"   : double},
                    "keypoints": [{"x": int ,  "y":  int ,  "v":  int}, 
                                {"x": int ,  "y": int ,  "v": int},
                                {"x":  int,  "y":  int,  "v": int} ] 
                }
    """
    image_path = os.path.join(main_path, "images", filename) + ".jpg"
    label_path = os.path.join(main_path, "labels", filename) + ".txt"

    image = cv2.imread(image_path)

    data = []
    with open(label_path, 'r') as file:
        for line in file:
            parts = list(map(float, line.strip().split()))
            if len(parts) != 14:
                raise ValueError(f"Invalid line format (expected 14 values, got {len(parts)}): {line.strip()}")

            dataset = {
                "class": int(parts[0]),
                "bbox": {
                    "x_center": parts[1],
                    "y_center": parts[2],
                    "width": parts[3],
                    "height": parts[4]
                },
                "keypoints": [
                    {"x": parts[5],  "y": parts[6],  "v": int(parts[7])},
                    {"x": parts[8],  "y": parts[9],  "v": int(parts[10])},
                    {"x": parts[11], "y": parts[12], "v": int(parts[13])}
                ]
            }

            data.append(dataset)

    return (image, data)



def show_image(image_plus_data, is_scaled = True, is_confidence = False):
    """
    show_image(tuple : image_plus_data, bool : is_scaled = True)
        * image_plus_data : (image, data)
        * is_scaled : whether the coordinates are scaled (0~640)
        - this function shows the labeled data on top of the image
        - you can choose if the data is scaled
    """
    image, data = image_plus_data

    if image is None:
        raise FileNotFoundError(f"Cannot load image")

    height, width = image.shape[:2]

    for object in data:
        
        bbox = object['bbox']

        if is_scaled:
            x_center = bbox['x_center'] * width
            y_center = bbox['y_center'] * height
            box_width = bbox['width'] * width
            box_height = bbox['height'] * height
        else:
            x_center = bbox['x_center']
            y_center = bbox['y_center'] 
            box_width = bbox['width']
            box_height = bbox['height'] 

        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            image,
            text = str(round(bbox["confidence"],2)),
            org = (x1, y1 - 5),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.4,       # 👈 smaller scale = smaller text
            color=(0,0,0),      # Black text
            thickness=1,
            lineType=cv2.LINE_AA
        )


        color_map = [(255,255,255), (255, 255, 0), (255, 0, 255)]

        for i, kp in enumerate(object['keypoints']):
            
            if is_scaled:
                x = int(kp['x'] * width)
                y = int(kp['y'] * height)
            else:
                x = int(kp['x'])
                y = int(kp['y'])

            v = kp['v']

            if v == 0: continue
            if v == 1:
                cv2.drawMarker(
                    image,
                    position=(x, y),
                    color=color_map[i],     
                    markerType=cv2.MARKER_TILTED_CROSS,
                    markerSize=6,
                    thickness=2
                )
                if is_confidence == True:
                    cv2.putText(
                        image,
                        text = str(round(kp['confidence'],2)),
                        org = (x + 5, y - 5),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.4,       # 👈 smaller scale = smaller text
                        color=color_map[i],      # Black text
                        thickness=1,
                        lineType=cv2.LINE_AA
                    )
            if v == 2:
                cv2.circle(
                    image,
                    center=(x,y),
                    radius=4,
                    color=color_map[i],
                    thickness=-1)
                if is_confidence == True:
                    cv2.putText(
                        image,
                        text = str(round(kp['confidence'],2)),
                        org = (x + 5, y - 5),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.4,       # 👈 smaller scale = smaller text
                        color=color_map[i],      # Black text
                        thickness=1,
                        lineType=cv2.LINE_AA
                    )


    
    cv2.imshow("testing_image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def prediction_handling(results):
    """
    prediction_handling(yolo_predict : results) -> confidence_data
        * results : a yolo_predict type
        * confidence_data: list of the following dictionary  [ {} , {} , {} ]
                confidence_data = {
                    "class" : 0,
                    "bbox"  : { "x_center" : double,
                            "y_center" : double,
                            "width"    : double,
                            "height"   : double
                            "confidence" : double},
                    "keypoints": [{"x": int ,  "y":  int ,  "v":  int, "confidence": double}, 
                                {"x": int ,  "y": int ,  "v": int, "confidence": double },
                                {"x":  int,  "y":  int,  "v": int, "confidence": double } ] 
                }

    """
    confidence_data = []

    for result in results:  # each image/frame
        boxes = result.boxes
        kpts_xy = result.keypoints.xy  # shape: (num_instances, num_kpts, 2)
        kpts_conf = result.keypoints.conf  # shape: (num_instances, num_kpts)

        for i in range(len(boxes)):
            # ----- Bounding Box -----
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            conf = float(boxes.conf[i])

            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1

            bbox = {
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
                "confidence": conf
            }

            # ----- Keypoints -----
            keypoints = []
            for j in range(kpts_xy.shape[1]):
                x, y = kpts_xy[i][j]
                kp_conf = kpts_conf[i][j].item()
                v = 2 if (kp_conf > 0.5) else (1 if (kp_conf > 0.1) else 0)  # visibility flag (custom rule)

                keypoints.append({
                    "x": int(x.item()),
                    "y": int(y.item()),
                    "v": v,
                    "confidence": kp_conf
                })

            confidence_data.append({
                "class": 0,
                "bbox": bbox,
                "keypoints": keypoints
            })
 
    return confidence_data



def print_data(dddata):
    """
    print_data(confidence_data : dddata)
        - currently only works for confidence_data
    """
    def round_numbers(obj, ndigits=2):
        if isinstance(obj, dict):
            return {k: round_numbers(v, ndigits) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [round_numbers(v, ndigits) for v in obj]
        elif isinstance(obj, (int, float)):
            return round(obj, ndigits)
        else:
            return obj  # leave strings, None, etc. unchanged
    dddata = round_numbers(dddata, ndigits=3)
    
    for i, ddata in enumerate(dddata):
        data= ddata["bbox"]
        ata = ddata["keypoints"]
        print(f"Data #{i}| ")
        print(f"Box : ({data['x_center']}, {data['y_center']}, {data['width']}, {data['height']})")
        for i in range(3):
            ta = ata[i]
            print(f"Keypoints{i} : ({ta['x']}, {ta['y']}, {ta['v']})")
        print(f"Confidence : Box({data['confidence']}), Keypoints({ata[0]['confidence']}, {ata[1]['confidence']}, {ata[2]['confidence']})")



