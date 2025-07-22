import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from funcs import check_data, make_filenames, load_file, show_image, prediction_handling, print_data

path = "TEST3/Strawberry Feature Points.v10i.yolov8"

train_path = os.path.join(path, "train")
test_path = os.path.join(path, "test")
valid_path = os.path.join(path, "valid")

# List all files - this exlcudes the extension ()
train_filenames = make_filenames(train_path)
test_filenames = make_filenames(test_path)
valid_filenames = make_filenames(valid_path)

# Check Model Performance
from ultralytics import YOLO
model_path = 'TEST3/resultsresults3/runs/pose/yolov8n_strawberry7/weights/best.pt'
model = YOLO(model_path)

for file in test_filenames:
    res = load_file(test_path, file)
    results = model.predict(source=res[0], conf=0.5, save=False, verbose = False)
    results = prediction_handling(results)
    print_data(results)
    
    show_image((res[0], results), is_scaled = False, is_confidence = True)

    input("Press Enter to continue...")

