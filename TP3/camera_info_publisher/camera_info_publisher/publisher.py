import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
import yaml

class StereoCameraPublisher(Node):
    def __init__(self):
        super().__init__('stereo_camera_publisher')

        # Publishers
        self.left_image_pub = self.create_publisher(Image, '/left/image_raw', 10)
        self.right_image_pub = self.create_publisher(Image, '/right/image_raw', 10)
        self.left_info_pub = self.create_publisher(CameraInfo, '/left/camera_info', 10)
        self.right_info_pub = self.create_publisher(CameraInfo, '/right/camera_info', 10)

        # Subscribers a las imágenes originales
        self.create_subscription(Image, '/cam0/image_raw', self.left_image_callback, 10)
        self.create_subscription(Image, '/cam1/image_raw', self.right_image_callback, 10)

        # Cargar calibraciones
        self.left_camera_info = self.load_camera_info(
            '/home/sco/roboticaMovil/TP3/calibrationData/left.yaml',
            frame_id='left_camera_frame'
        )
        self.right_camera_info = self.load_camera_info(
            '/home/sco/roboticaMovil/TP3/calibrationData/right.yaml',
            frame_id='right_camera_frame'
        )

        self.get_logger().info('StereoCameraPublisher inicializado correctamente.')

    def load_camera_info(self, file_path, frame_id):
        """Carga un archivo YAML de calibración y lo convierte a un mensaje CameraInfo."""
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

    def left_image_callback(self, msg: Image):
        """Callback para la cámara izquierda."""
        # Actualiza el timestamp del CameraInfo y publica ambos
        self.left_camera_info.header.stamp = msg.header.stamp
        self.left_info_pub.publish(self.left_camera_info)
        self.left_image_pub.publish(msg)

    def right_image_callback(self, msg: Image):
        """Callback para la cámara derecha."""
        # Actualiza el timestamp del CameraInfo y publica ambos
        self.right_camera_info.header.stamp = msg.header.stamp
        self.right_info_pub.publish(self.right_camera_info)
        self.right_image_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = StereoCameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
