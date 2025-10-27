import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud, PointField, PointCloud2
from std_msgs.msg import Header
import yaml
import numpy as np
import os




class PointsCloudPublisher(Node):
    def __init__(self):
        super().__init__('point_cloud_publisher')


        self.point_cloud_dir = '/home/bruno/roboticaMovil/TP3/point_clouds/'
        self.files = sorted([
            os.path.join(self.point_cloud_dir, f) for f in os.listdir(self.point_cloud_dir) if f.endswith('.npy')
        ])
        self.current_file_index = 0

        self.pcd_publisher = self.create_publisher(PointCloud2, 'point_cloud', 10)
        timer_period = 0.5  # segundos
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        if not self.files:
            self.get_logger().warn('No point cloud files found in the directory.')
            return

        # Load the current point cloud file
        file_path = self.files[self.current_file_index]
        self.get_logger().info(f'Publishing point cloud from file: {file_path}')
        points = np.load(file_path)

        # Convert to PointCloud2 message
        self.pcd = array_to_point_cloud(points, 'map')
        self.pcd_publisher.publish(self.pcd)

        self.current_file_index = (self.current_file_index + 1) % len(self.files)

def array_to_point_cloud(points, parent_frame):
    header = Header()
    header.stamp = rclpy.time.Time().to_msg()
    header.frame_id = parent_frame

    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]

    point_cloud_msg = PointCloud2()
    point_cloud_msg.header = header
    point_cloud_msg.height = 1
    point_cloud_msg.width = points.shape[0]
    point_cloud_msg.fields = fields
    point_cloud_msg.is_bigendian = False
    point_cloud_msg.point_step = 12
    point_cloud_msg.row_step = point_cloud_msg.point_step * points.shape[0]
    point_cloud_msg.is_dense = True
    point_cloud_msg.data = np.asarray(points, np.float32).tobytes()

    return point_cloud_msg

def main(args=None):
    rclpy.init(args=args)
    node = PointsCloudPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
