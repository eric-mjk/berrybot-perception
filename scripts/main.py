import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from funcs import check_data, make_filenames, load_file, show_image

# Target Directory
path = "TEST4/masked_dataset.yolov8"

train_path = os.path.join(path, "train")
test_path = os.path.join(path, "test")
valid_path = os.path.join(path, "valid")

# List all files - this exlcudes the extension ()
train_filenames = make_filenames(train_path)
test_filenames = make_filenames(test_path)
valid_filenames = make_filenames(valid_path)


# Sample
image, data = load_file(train_path, train_filenames[0])
# print(data)

