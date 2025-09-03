# YOLO-pose Model for Fruit Detection
![val_batch2_pred](https://github.com/user-attachments/assets/e99de61f-8a4a-41b5-8ae5-a7d8840da6ff)

## Overview
This is a workspace for 2025 Seoul National University Creative Engineering Fair (2025 서울대학교 창의설계축전).
This repository contains a fine-tuned version of the Ultralytics YOLO-Pose v11 model, specifically adapted for the berrybot project. The model has been re-trained on a custom dataset to improve keypoint detection for our specific application.

## Key Changes from the Ultralytics YOLO-Pose v11
While this project is built upon the robust foundation of the Ultralytics YOLO-Pose v11 framework, some modifications were made to optimize performance for our custom dataset:

- Modified Object Keypoint Similarity (OKS) Metric: The OKS metric, a key measure of keypoint detection accuracy, was adjusted to better suit the specific characteristics and scale of the objects in our dataset. This fine-tuning of the OKS calculation ensures more precise evaluation and training feedback.
<img width="468" height="647" alt="Feature Point Label" src="https://github.com/user-attachments/assets/fda34a82-a8f6-4f7c-9fae-417afd66c021" />
From the above, there are 3 keypoints in our dataset. The three keypoints were assigned the following OKS Values [Bottom : 0.05, Top : 0.05, Pick : 0.1]

- Parameter Tuning: The training and detection parameters of the model were extensively fine-tuned. Modifications were mainly on the calculation of the Loss Function. Also some parameters relate to training (lr, epoch, patience) where modified.
The parameters can be found at 'args.yaml'.


## Clone the repository:

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
