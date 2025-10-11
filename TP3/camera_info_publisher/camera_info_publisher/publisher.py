import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
import yaml

class StereoCameraInfoPublisher(Node):
    def __init__(self):
        super().__init__('stereo_camera_info_publisher')

        # Publishers
        self.left_camera_pub = self.create_publisher(CameraInfo, '/left/camera_info', 10)
        self.right_camera_pub = self.create_publisher(CameraInfo, '/right/camera_info', 10)

        # Subscribers to image_raw topics
        self.create_subscription(Image, '/stereo/left/image_raw', self.left_image_callback, 10)
        self.create_subscription(Image, '/stereo/right/image_raw', self.right_image_callback, 10)

        # Load YAML calibration files
        self.left_camera_info = self.load_camera_info(
            '/home/bruno/roboticaMovil/TP3/calibrationData/left.yaml',
            frame_id='left_camera_frame'
        )
        self.right_camera_info = self.load_camera_info(
            '/home/bruno/roboticaMovil/TP3/calibrationData/right.yaml',
            frame_id='right_camera_frame'
        )
    def load_camera_info(self, file_path, frame_id):
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)

        camera_info = CameraInfo()
        camera_info.header.frame_id = frame_id
        camera_info.width = data['image_width']
        camera_info.height = data['image_height']
        camera_info.k = data['camera_matrix']['data']
        camera_info.d = data['distortion_coefficients']['data']
        camera_info.r = data['rectification_matrix']['data']
        camera_info.p = data['projection_matrix']['data']
        camera_info.distortion_model = data['distortion_model']
        return camera_info

    def left_image_callback(self, msg):
        # Update timestamp and publish left CameraInfo
        self.left_camera_info.header.stamp = msg.header.stamp
        self.left_camera_pub.publish(self.left_camera_info)

    def right_image_callback(self, msg):
        # Update timestamp and publish right CameraInfo
        self.right_camera_info.header.stamp = msg.header.stamp
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