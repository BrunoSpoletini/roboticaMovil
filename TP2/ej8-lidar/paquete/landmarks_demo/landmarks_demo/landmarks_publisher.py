import rclpy
from rclpy.node import Node

from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import LaserScan

class LandmarksPublisher(Node):
    def __init__(self):
        super().__init__('landmarks_publisher')

        # Frame en el que “viven” los landmarks
        self.frame_id = 'base_scan'

        self.marker_pub = self.create_publisher(MarkerArray, 'landmarks/markers', 10)

        # Timer
        self.timer = self.create_timer(0.5, self.publish_landmarks)

        self.landmarks = []
        try:
            with open('../../landmarks.txt', 'r') as f:
                for i, line in enumerate(f):
                    x, y, r = map(float, line.strip().split(','))
                    self.landmarks.append((i, x, y, r))
            self.get_logger().info(f'Leídos {len(self.landmarks)} landmarks de landmarks.txt')
        except Exception as e:
            self.get_logger().error(f'Error leyendo landmarks.txt: {e}')

    def publish_landmarks(self):

        # MarkerArray (para verlos en RViz2)
        ma = MarkerArray()
        now = self.get_clock().now().to_msg()

        

        for (i, (lid, x, y, r)) in enumerate(self.landmarks):
            # Cilindro
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = now
            m.ns = 'landmarks'
            m.id = lid
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = r / 2.0 
            m.pose.orientation.w = 1.0
            m.scale.x = r * 2.0             # diámetro X
            m.scale.y = r * 2.0             # diámetro Y
            m.scale.z = r                   # alto del cilindro
            # color RGBA (0..1)
            m.color.r = 0.1
            m.color.g = 0.6
            m.color.b = 1.0
            m.color.a = 0.9
            m.lifetime.sec = 0
            ma.markers.append(m)
        self.marker_pub.publish(ma)

def main():
    rclpy.init()
    node = LandmarksPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
