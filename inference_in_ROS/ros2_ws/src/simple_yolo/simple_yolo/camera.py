#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class Camera(Node):
    def __init__(self):
        super().__init__('camera')
        self.image_publisher_ = self.create_publisher(Image, 'image', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz
        self.cap = cv2.VideoCapture(0)
        self.bridge = CvBridge()

        if not self.cap.isOpened():
            self.get_logger().error('Failed to open camera.')

    def timer_callback(self):
        ret, frame = self.cap.read()

        if not ret:
            self.get_logger().warn('Failed to read frame')
            return

        frame = cv2.resize(frame, (640, 640))
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.image_publisher_.publish(msg)
        self.get_logger().info('Publishing frame')

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = Camera()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
