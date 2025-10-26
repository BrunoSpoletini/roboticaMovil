import rclpy
import cv2
import argparse
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped

# Nodo encargado de escuhcar las imagenes crudas
class RawImagesListener(Node):
    def __init__(self, outputDir):
        super().__init__('ground_truth_listener')
        self.bridge = CvBridge()
        # Variables de estado
        self.index = 0
        self.poses = []
        self.outputDir = outputDir
        # Creamos subcriptores para almacenar las imagenes crudas
        self.sub = self.create_subscription(Image, '/cam0/image_raw', self.saveImageCallback, 10)
        self.sub = self.create_subscription(Image, '/cam1/image_raw', self.saveImageCallback, 10)

    # Almacena la imagen cruda en un archivo
    def saveImageCallback(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv2.imwrite(f"{self.outputDir}/left_{self.index:04d}.png", cv_img)
        self.index += 1

# Parsea los argumentos de entrada
def parse_args():
    parser = argparse.ArgumentParser(description="Directorio donde las imagenes crudas del ground-truth")
    parser.add_argument("--o", type=str, default="RawImages", help="Directorio donde guardar las imagenes crudas")
    return parser.parse_args()

def main(args=None):
    # Parseamos los argumentos
    outputDir = parse_args()

    # Inicializamoos el nodo
    rclpy.init(args=args)
    node = RawImagesListener(outputDir)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
