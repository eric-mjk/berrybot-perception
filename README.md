# berrybot-perception
This repository contains a fine-tuned version of the Ultralytics YOLO-Pose v11 model, specifically adapted for the berrybot project. The model has been re-trained on a custom dataset to improve keypoint detection for our specific application. This README outlines the model's features, the key modifications from the original implementation, and the repository's structure.

Key Changes from the Ultralytics YOLO-Pose v11
While this project is built upon the robust foundation of the Ultralytics YOLO-Pose v11 framework, several modifications were made to optimize performance for our custom dataset:

Modified Object Keypoint Similarity (OKS) Metric: The OKS metric, a key measure of keypoint detection accuracy, was adjusted to better suit the specific characteristics and scale of the objects in our dataset. This fine-tuning of the OKS calculation ensures more precise evaluation and training feedback.

Parameter Tuning: The training and detection parameters of the model were extensively fine-tuned. This includes adjustments to learning rates, anchor box sizes, and confidence thresholds to maximize the model's performance on our unique data, leading to a more specialized and accurate model for our use case.

Repository Structure
This repository follows a straightforward structure to make it easy to navigate and use.

berrybot-perception/
├── data/
│   ├── images/         # Custom dataset images
│   └── labels/         # Keypoint annotations
├── models/
│   └── custom_yolov11.pt # Fine-tuned model weights
├── src/
│   ├── train.py      # Script used for fine-tuning
│   └── detect.py     # Script for running inference on new data
├── README.md           # This file
└── requirements.txt    # Required Python packages

Getting Started
To get started with this project, we recommend familiarizing yourself with the original Ultralytics documentation.

Clone the repository:

git clone [https://github.com/your-username/berrybot-perception.git](https://github.com/your-username/berrybot-perception.git)
cd berrybot-perception

Install dependencies:

pip install -r requirements.txt

Train the model:
You can use the train.py script to fine-tune the model further with your own data.

License
This project is licensed under the MIT License.

Credits
Ultralytics: For creating and maintaining the YOLO-Pose models, which serve as the foundation for this work.
