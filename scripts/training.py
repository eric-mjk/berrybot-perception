import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from funcs import check_data, make_filenames, load_file, show_image

# from custom_imports import custom_import
# custom_import()

import sys
sys.path.insert(0, r"C:/Users/user/Desktop/User_Eric/git_folder/ultralytics")
import ultralytics
print(ultralytics.__file__)     # check the source
print(ultralytics.__version__)

import ultralytics
import torch
from ultralytics import YOLO

# Target Directory
path = "Strawberry Feature Points.v11-complete_dataset.yolov8/data.yaml"

1 ###############################################################################
model = YOLO("yolov8s-pose.yaml")  # build a new model from YAML

model.train(
    data= path,
    epochs=100,
    imgsz=640,
    batch=16,
    workers=2,
    name='yolov8n_strawberry_8s',
    project='C:/Users/user/Desktop/User_Eric/data',
)


2 ###############################################################################
model = YOLO("yolo11s-pose.yaml")  # build a new model from YAML

model.train(
    data= path,
    epochs=100,
    imgsz=640,
    batch=16,
    workers=2,
    name='yolov8n_strawberry_11s',
    project='C:/Users/user/Desktop/User_Eric/data',
)

3 ###############################################################################
model = YOLO("yolov8s-pose.pt")  # build a new model from YAML

model.train(
    data= path,
    epochs=100,
    imgsz=640,
    batch=16,
    workers=2,
    name='yolov8n_strawberry_pretrained8s',
    project='C:/Users/user/Desktop/User_Eric/data',
)

4 ###############################################################################
model = YOLO("yolo11s-pose.pt")  # build a new model from YAML

model.train(
    data= path,
    epochs=100,
    imgsz=640,
    batch=16,
    workers=2,
    name='yolov8n_strawberry_pretrained8s',
    project='C:/Users/user/Desktop/User_Eric/data',
)

