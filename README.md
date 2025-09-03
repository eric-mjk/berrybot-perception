# Overview
This is a repository is for 2025 Seoul National University Creative Engineering Fair (2025 서울대학교 창의설계축전).

- Team : Berry-Good-Bot (딸기가 좋아)

- Robot : Fruit Harvesting Robot (과일 수확 로봇)

This repository only contains the "perception" related tasks of the robot. More information on the robot can be found below.
- ROS-Control + Overview on Project : https://github.com/sawo0150/berry-good-bot
- Presentation Video                : https://www.youtube.com/watch?v=eAwnoZv1Vwg
- Robot Operation Video             : https://www.youtube.com/watch?v=T6ebTg6ipFc

<img width="800" height="600" alt="과일수확로봇-대표이미지" src="https://github.com/user-attachments/assets/0f91454a-fdf4-4247-98a1-81b32c4a3944" />

# YOLO-pose Model for Fruit Detection
<img width="600" height="600" alt="val_batch2_pred" src="https://github.com/user-attachments/assets/e99de61f-8a4a-41b5-8ae5-a7d8840da6ff" />

This repository contains a fine-tuned version of the Ultralytics YOLO-Pose v11 model, specifically adapted for the Fruit Harvesting Robot project. The model has been re-trained on a custom dataset to improve keypoint detection for our specific application.

## Dataset
The dataset can be found at the following ROBOFLOW workspace. It is consisted with 900 images of 2500 individual fruits. It is labeled with a bounding box and three feature points.

https://universe.roboflow.com/berrybot-xzypx/fruit-pose-fdtan

<img width="600" height="600" alt="Label Summary" src="https://github.com/user-attachments/assets/95fef41f-8088-4dbd-b862-2b39fac4909a" />

(Label Summary)

## Key Changes from the Ultralytics YOLO-Pose v11
While this project is built upon the robust foundation of the Ultralytics YOLO-Pose v11 framework, some modifications were made to optimize performance for our custom dataset:

- Modified Object Keypoint Similarity (OKS) Metric: The OKS metric, a key measure of keypoint detection accuracy, was adjusted to better suit the specific characteristics and scale of the objects in our dataset. This fine-tuning of the OKS calculation ensures more precise evaluation and training feedback. From the image below, there are 3 keypoints in our dataset. The three keypoints were assigned the following OKS Values [Bottom : 0.05, Top : 0.05, Pick : 0.1]
<img width="200" height="400" alt="Feature Point Label" src="https://github.com/user-attachments/assets/fda34a82-a8f6-4f7c-9fae-417afd66c021" />

- Parameter Tuning: The training and detection parameters of the model were extensively fine-tuned. Modifications were mainly on the calculation of the Loss Function. Also some parameters relate to training (lr, epoch, patience) where modified.
The parameters can be found at 'parameters' folder.


## Model Performance
Model Performance is based on 'result2'. 'result1' and 'result2' has almost identical performance, however we suggest 'result2' model works a bit better in real-time.

<img width="400" height="400" alt="BoxPR_curve" src="https://github.com/user-attachments/assets/4b0cc6df-8dd8-4d2b-90b9-baa3c8b10f6f" />
<img width="400" height="400" alt="PosePR_curve" src="https://github.com/user-attachments/assets/fe4fab33-1c25-4c58-8ce8-4919c9e6d94e" />

(Precision-Recall curve of 'Bounding Box' and 'Pose(Feature Point)')

<img width="4200" height="600" alt="results" src="https://github.com/user-attachments/assets/d62e8dd4-51ac-469c-81fa-3358ae333e7e" />
(Model training results)

## Repository Structure
```
berrybot-perception/
├── models/             # Fine-tuned model weights for each case
│   ├── result1.pt 
│   └── result2.pt
├── parameters/         # Model parameters for each case
│   ├── result1.yaml
│   └── result2.yaml
├── scripts/            # Scripts for training and inference of model
│   └── ...
├── inference_in_ROS/   # Files realted to use the YOLO model for Perception in ROS enviroment
│   └── ...
└── README.md           # This file
```

## Contributors
Taeyoung Kwon(권태영)* : Robot Hardware Design and Manufacturing

Minjun Kim(김민준)* : YOLO-Pose Deep Learning Model Development

Sangwon Park(박상원)* : System Integration and Control Software Development

(*Seoul National University, Department of Mechanical Engineering)
