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

###############################################################################
from ultralytics import YOLO
model = YOLO("yolov8s-pose.yaml")  # build a new model from YAML

model.train(
    data='/content/drive/MyDrive/BerryBot/TESTS/4. Test4/masked_dataset.yolov8/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    workers=2,
    name='yolov8n_strawberry',
)


# Target Directory
path = "Strawberry Feature Points.v11-complete_dataset.yolov8"


