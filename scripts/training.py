# training.py

# basic imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys

from funcs import check_data, make_filenames, load_file, show_image

# custom imports
sys.path.insert(0, r"C:/Users/user/Desktop/User_Eric/git_folder/ultralytics")
import ultralytics
print(ultralytics.__file__)     # check the source
print(ultralytics.__version__)

# setting and paths
from ultralytics import YOLO
path = "revised_dataset.yolov8\\data.yaml"


if __name__ == '__main__':

    1 ###############################################################################
    # trying 'nano'
    model = YOLO("yolo11n-pose.pt")  # build a new model from YAML

    model.train(
        data= path,
        epochs=300,
        imgsz=640,
        batch=16,
        workers=4,
        name='(ES = True, OKS = [0.05, 0.02, 0.07], Box Loss = 1.0) 11nano-pt',
        project='C:/Users/user/Desktop/User_Eric/data',
        patience=50
    )