"""
SyncNode: Synchronizes image data with driving info and publishes a fused custom message.

Subscribes:
- /camera/image_raw (sensor_msgs/Image): Incoming camera stream
- /driving_info (geometry_msgs/Vector3): Driving data (steering angle, lane offset)

Publishes:
- /synced_output (custom_msgs/SyncedOutput): Fused message with image, steering, lane offset, and timestamp

Note: This assumes that `SyncedOutput.msg` includes:
- sensor_msgs/Image image
- float64 steering_angle
- float64 lane_offset
- builtin_interfaces/Time camera_time
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3
from cv_bridge import CvBridge
from collections import deque
from custom_msgs.msg import SyncedOutput  # Custom message definition


class SyncNode(Node):
    """
    A ROS 2 node that synchronizes camera images with driving information and publishes
    a combined message containing both streams' data.

    Attributes:
        bridge (CvBridge): Converts ROS images to OpenCV format (not used directly here).
        image_buffer (deque): Stores recent image messages with reception timestamps.
        latest_driving_data (dict): Holds the latest steering angle and lane offset.
    """

    def __init__(self):
        """
        Initializes the SyncNode with required subscriptions and a publisher.
        """
        super().__init__('sync_node')
        self.bridge = CvBridge()

        self.image_buffer = deque(maxlen=10)  # Stores (Image msg, reception time)
        self.latest_driving_data = None       # Stores latest driving info

        # Subscriptions
        self.create_subscription(Image, '/camera/image_raw', self.camera_callback, 10)
        self.create_subscription(Vector3, '/driving_info', self.driving_callback, 10)

        # Publisher
        self.publisher = self.create_publisher(SyncedOutput, '/synced_output', 10)

        self.get_logger().info("Sync Node with Vector3 driving info is running.")

    def camera_callback(self, msg):
        """
        Callback for receiving camera images.

        Args:
            msg (sensor_msgs.msg.Image): Incoming camera frame.

        Adds the image and its reception time (PC clock) to a buffer for later synchronization.
        """
        now = self.get_clock().now().to_msg()
        self.image_buffer.append((msg, now))

    def driving_callback(self, msg):
        """
        Callback for receiving driving info data (Vector3).

        Args:
            msg (geometry_msgs.msg.Vector3): Contains `steering_angle` and `lane_offset`.

        Combines the latest image and driving info into a `SyncedOutput` message
        and publishes it.
        """
        # Update stored driving data
        self.latest_driving_data = {
            'steering_angle': msg.x,  # Assuming x = steering_angle
            'lane_offset': msg.y      # Assuming y = lane_offset
        }

        if not self.image_buffer:
            self.get_logger().warn("No image available to sync with driving info.")
            return

        # Use latest image
        image_msg, image_time = self.image_buffer[-1]

        # Construct fused message
        fused_msg = SyncedOutput()
        fused_msg.image = image_msg
        fused_msg.steering_angle = self.latest_driving_data['steering_angle']
        fused_msg.lane_offset = self.latest_driving_data['lane_offset']
        fused_msg.camera_time = image_time

        # Publish synced output
        self.publisher.publish(fused_msg)
        self.get_logger().info("Published synced data.")


def main(args=None):
    """
    Initializes the ROS 2 system and starts the SyncNode.
    """
    rclpy.init(args=args)
    node = SyncNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
