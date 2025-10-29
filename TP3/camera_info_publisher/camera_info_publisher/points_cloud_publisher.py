import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np
import os

class PointCloudPublisher(Node):
    def __init__(self, input_dir, scale=1.0):
        super().__init__('point_cloud_publisher')

        self.input_dir = input_dir
        self.scale = scale


        self.sub_points_cloud = 
            self.create_subscription(PointCloud2,'/Publisher/filtered_points',self.instancia_nube_puntos_callback,10)

        self.pub = self.create_publisher(PointCloud2, 'colored_point_cloud', 10)


    def instancia_nube_puntos_callback(self, msg):


        self.get_logger().info(f'Publishing {file_path}')
        points = msg

        # Transformar a mundo y filtrar NaNs
        if points.ndim == 3:
            pts = points.reshape(-1, 3)
        else:
            pts = points
        mask = ~np.isnan(pts).any(axis=1)
        pts = pts[mask]

        # Aplicar escala si se pasó
        pts = self.scale * pts

        msg = self.points_to_pointcloud2(pts, frame_id='map')
        self.pub.publish(msg)

        self.index = (self.index + 1) % len(self.files)


    @staticmethod
    def points_to_pointcloud2(points, frame_id='map'):
        header = Header()
        header.stamp = rclpy.time.Time().to_msg()
        header.frame_id = frame_id

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = points.shape[0]
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * points.shape[0]
        msg.is_dense = True
        msg.data = points.astype(np.float32).tobytes()

        return msg

def main():
    import sys
    if len(sys.argv) < 2:
        print("Uso: ros2 run <package> pointcloud_publisher.py <directorio_npy>")
        return

    input_dir = sys.argv[1]
    scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    rclpy.init()
    node = PointCloudPublisher(input_dir, scale)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
