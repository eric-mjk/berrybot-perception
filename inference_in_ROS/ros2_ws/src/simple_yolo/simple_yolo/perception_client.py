#!/usr/bin/env python3

import sys
import select
import termios
import tty

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from interfaces.action import Perception


def get_single_key(timeout: float = 0.1) -> str:
    """Read one character from stdin if available within timeout (non-blocking)."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        rlist, _, _ = select.select([fd], [], [], timeout)
        if rlist:
            return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ''


class PerceptionClient(Node):
    def __init__(self):
        super().__init__('perception_client')
        self._action_client = ActionClient(self, Perception, '/perception')
        self._goal_sent = False

        # Poll keyboard every 0.1s
        self.create_timer(0.1, self._check_for_space)

    def _check_for_space(self):
        key = get_single_key()
        if key == ' ' and not self._goal_sent:
            self.get_logger().info("Spacebar pressed → sending Perception goal")
            self.send_goal()
        elif key == ' ' and self._goal_sent:
            self.get_logger().warn("Goal already sent; waiting for result")

    def send_goal(self, run: bool = True):
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Action server not available!')
            return

        goal_msg = Perception.Goal()
        goal_msg.run = run

        send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)
        self._goal_sent = True

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal rejected')
            self._goal_sent = False
            return

        self.get_logger().info('Goal accepted')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def feedback_callback(self, feedback_msg):
        fb = feedback_msg.feedback
        self.get_logger().info(f'[Feedback] {fb.status}')

    def result_callback(self, future):
        res : Perception.Result = future.result().result
        detection = res.yolo_result

        self.get_logger().info('--- /perception Result ---')
        self.get_logger().info(
            f'bbox: x={detection.x:.3f}, y={detection.y:.3f}, '
            f'w={detection.w:.3f}, h={detection.h:.3f}'
        )
        self.get_logger().info(f'confidence: {detection.conf:.3f}')
        # If you have keypoints:
        self.get_logger().info(
            f'kp1: ({detection.x1:.3f},{detection.y1:.3f}); '
            f'kp2: ({detection.x2:.3f},{detection.y2:.3f}); '
            f'kp3: ({detection.x3:.3f},{detection.y3:.3f})'
        )
        
        # …log other fields if needed…
        rclpy.shutdown()


def main(args=None):
    # Put terminal into raw mode so we can read single keypresses
    rclpy.init(args=args)
    client = PerceptionClient()
    rclpy.spin(client)


if __name__ == '__main__':
    main()
