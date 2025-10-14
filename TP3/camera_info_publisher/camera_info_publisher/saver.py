import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
import cv2
import os

class StereoSaver(Node):
    def __init__(self):
        super().__init__('stereo_image_saver')
        self.bridge = CvBridge()
        self.counter = 0
        self.output_dir = '/home/sco/roboticaMovil/TP3/stereo_captures'
        os.makedirs(self.output_dir, exist_ok=True)

        # Suscriptores sincronizados
        left_sub = Subscriber(self, Image, '/left/image_rect')
        right_sub = Subscriber(self, Image, '/right/image_rect')

        sync = ApproximateTimeSynchronizer([left_sub, right_sub], queue_size=10, slop=0.1)
        sync.registerCallback(self.callback)

    def callback(self, left_msg, right_msg):
        left_img = self.bridge.imgmsg_to_cv2(left_msg, 'bgr8')
        right_img = self.bridge.imgmsg_to_cv2(right_msg, 'bgr8')

        left_path = os.path.join(self.output_dir, f'pair_{self.counter:04d}_left.png')
        right_path = os.path.join(self.output_dir, f'pair_{self.counter:04d}_right.png')

        cv2.imwrite(left_path, left_img)
        cv2.imwrite(right_path, right_img)

        self.get_logger().info(f'Saved pair {self.counter:04d}')
        self.counter += 1

def main(args=None):
    rclpy.init(args=args)
    node = StereoSaver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
