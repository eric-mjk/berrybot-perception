#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer

from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from interfaces.action import Perception  # Replace with your actual package
from interfaces.msg import YoloDetection
"""
# Goal definition
bool run
---
# Result definition
YoloDetection yolo_result
---
# Feedback definition
string status
"""
"""
YoloDetection
float32 x
float32 y
float32 w
float32 h
float32 conf
float32 x1
float32 y1
float32 conf1
float32 x2
float32 y2
float32 conf2
float32 x3
float32 y3
float32 conf3
"""

import os
import sys
from ament_index_python.packages import get_package_share_directory

from ultralytics import YOLO

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        model_path = os.path.join(
            get_package_share_directory('simple_yolo'),
            'models', 'model1.pt'
        )

        self.model = YOLO(model_path)
        self.bridge = CvBridge()
        self.latest_image = None
        self.latest_perception = None

        # Image subscriber
        self.subscription = self.create_subscription(
            Image,
            'image',
            self.image_callback,
            10
        )

        # Action server
        self.action_server = ActionServer(
            self,
            Perception,
            'perception',
            self.execute_callback
        )

        self.get_logger().info("Perception node initialized.")

    def image_callback(self, msg):
        self.latest_image = msg
        self.get_logger().debug("Image received.")

    def execute_callback(self, goal_handle):
        self.get_logger().info("Received perception goal.")

        if self.latest_image is None:
            self.get_logger().warn("No image available yet.")
            goal_handle.abort()
            return Perception.Result()

        # Convert to OpenCV image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            goal_handle.abort()
            return Perception.Result()

        # Placeholder processing function (res is type msg : YoloDetection)
        processed_model_result = self.process_image(cv_image)
        assert type(processed_model_result) == list
        if len(processed_model_result) == 0:
            self.get_logger().warn(f"No strawberries detected")
            goal_handle.abort()
            return Perception.Result()
        
        self.latest_perception = self.pick_perception(processed_model_result)
        result = Perception.Result(yolo_result = self.latest_perception)
        goal_handle.succeed()

        return result

    def process_image(self, cv_image):
        self.get_logger().info("Processing image...")
        model_result = self.model.predict(source=cv_image, conf=0.3, save=False, verbose=False)
        model_result = self.prediction_handling(model_result)
        return model_result


    def prediction_handling(self, results):
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


    def pick_perception(self, model_result) -> YoloDetection:
        """
        pick_perception(self, model_result)
        - this function will receive a processed model result (which is asserted to be a list of dictionaries & len != 0)
        - this function will return 1 perception result based on its previous perception results
        """
        max_conf = -1
        data = None
        for res in model_result:
            if max_conf < res["bbox"]["confidence"]:
                max_conf = res["bbox"]["confidence"]
                data = res

        det = YoloDetection()
        bbox = data["bbox"]
        kps = data["keypoints"]

        det.x = float(bbox["x_center"])
        det.y = float(bbox["y_center"])
        det.w = float(bbox["width"])
        det.h = float(bbox["height"])
        det.conf = float(bbox["confidence"])

        det.x1, det.y1, det.conf1 = float(kps[0]["x"]), float(kps[0]["y"]), float(kps[0]["confidence"])
        det.x2, det.y2, det.conf2 = float(kps[1]["x"]), float(kps[1]["y"]), float(kps[1]["confidence"])
        det.x3, det.y3, det.conf3 = float(kps[2]["x"]), float(kps[2]["y"]), float(kps[2]["confidence"])
        
        return det



def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
