import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class ImageSaver(Node):
    def __init__(self, output_dir):
        super().__init__('image_saver')
        self.bridge = CvBridge()

        # Crear directorio de salida
        self.output_dir = output_dir
        os.makedirs(os.path.join(output_dir, "left"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "right"), exist_ok=True)

        # Índices de imagen guardadas
        self.saved_left = 0
        self.saved_right = 0

        # Suscribirse a los tópicos de las cámaras
        self.sub_cam0 = self.create_subscription(
            Image,
            '/cam0/image_raw',
            self.callback_cam0,
            10
        )
        self.sub_cam1 = self.create_subscription(
            Image,
            '/cam1/image_raw',
            self.callback_cam1,
            10
        )

        # Para evitar guardar duplicados
        self.last_stamp_cam0 = None
        self.last_stamp_cam1 = None

    def callback_cam0(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp == self.last_stamp_cam0:
            return  # Ya la guardamos
        self.last_stamp_cam0 = stamp

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        filename = os.path.join(self.output_dir, "cam0", f"img_{self.saved_left:06d}.png")
        cv2.imwrite(filename, cv_image)
        self.saved_left += 1
        self.get_logger().info(f"💾 Guardada {filename}")

    def callback_cam1(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp == self.last_stamp_cam1:
            return  # Ya la guardamos
        self.last_stamp_cam1 = stamp

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        filename = os.path.join(self.output_dir, "cam1", f"img_{self.saved_right:06d}.png")
        cv2.imwrite(filename, cv_image)
        self.saved_right += 1
        self.get_logger().info(f"💾 Guardada {filename}")

def main(args=None):
    import sys
    if len(sys.argv) < 2:
        print("Uso: ros2 run camera_info_publisher save_images_node.py <directorio_salida>")
        return

    output_dir = sys.argv[1]
    
    rclpy.init(args=args)
    node = ImageSaver(output_dir)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
