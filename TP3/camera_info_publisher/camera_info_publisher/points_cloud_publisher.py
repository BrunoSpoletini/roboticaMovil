import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud, PointField, PointCloud2
from std_msgs.msg import Header
import yaml
import numpy as np

class PointsCloudPublisher(Node):
    def __init__(self):
        super().__init__('point_cloud_publisher')

        # leemos de un archivo .npy los puntos 3D
        self.points = np.load('/home/sco/roboticaMovil/TP3/output.npy', allow_pickle=True)

        self.pcd_publisher = self.create_publisher(PointCloud2, 'point_cloud_ejd', 10)
        timer_period = 1/30.0
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        self.pcd = array_to_point_cloud(self.points, 'map')
        self.pcd_publisher.publish(self.pcd)

def array_to_point_cloud(points, parent_frame):
    # Asegurarse que los puntos sean float32
    points = points.astype(np.float32)

    # Verificar que points tenga forma (H, W, 3)
    if len(points.shape) != 3 or points.shape[2] != 3:
        raise ValueError(f"Formato inválido de puntos: {points.shape}. Se esperaba (H, W, 3).")

    h, w, _ = points.shape

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
    point_cloud_msg.height = h
    point_cloud_msg.width = w
    point_cloud_msg.fields = fields
    point_cloud_msg.is_bigendian = False
    point_cloud_msg.point_step = 12  # 3 * float32
    point_cloud_msg.row_step = point_cloud_msg.point_step * w
    point_cloud_msg.is_dense = True
    point_cloud_msg.data = points.tobytes()

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