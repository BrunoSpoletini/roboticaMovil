import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
import yaml

class StereoCameraInfoPublisher(Node):
    def __init__(self):
        super().__init__('stereo_camera_info_publisher')

        # Publishers for left and right camera info
        self.left_camera_pub = self.create_publisher(CameraInfo, '/left/camera_info', 10)
        self.right_camera_pub = self.create_publisher(CameraInfo, '/right/camera_info', 10)

        # Timer to publish at 10 Hz
        self.timer = self.create_timer(0.1, self.publish_camera_info)

        # Load calibration data from YAML files
        self.left_camera_info = self.load_camera_info('/home/bruno/roboticaMovil/TP3/calibrationData/left.yaml')
        self.right_camera_info = self.load_camera_info('/home/bruno/roboticaMovil/TP3/calibrationData/right.yaml')

    def load_camera_info(self, file_path):
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)

        camera_info = CameraInfo()
        camera_info.header.frame_id = data.get('camera_name', 'camera_frame')
        camera_info.width = data['image_width']
        camera_info.height = data['image_height']
        camera_info.k = data['camera_matrix']['data']
        camera_info.d = data['distortion_coefficients']['data']
        camera_info.distortion_model = data['distortion_model']
        return camera_info

    def publish_camera_info(self):
        # Update timestamps
        self.left_camera_info.header.stamp = self.get_clock().now().to_msg()
        self.right_camera_info.header.stamp = self.get_clock().now().to_msg()

        # Publish camera info messages
        self.left_camera_pub.publish(self.left_camera_info)
        self.right_camera_pub.publish(self.right_camera_info)

def main(args=None):
    rclpy.init(args=args)
    node = StereoCameraInfoPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()