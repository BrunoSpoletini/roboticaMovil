import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

# Nodo encargado de publicar la posicion 
class TrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('trajectory_publisher')

        self.path_pub = self.create_publisher(Path, '/camera_trajectory', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.poses_gt = np.load('poses_l.npy')
        self.index = 0

        self.path_msg = Path()
        self.path_msg.header.frame_id = "map"

    def timer_callback(self):
        if self.index >= len(self.poses_gt):
            return

        pose = self.poses_gt[self.index]
        t = pose[:3]
        q = pose[3:]

        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "map"
        pose_stamped.header.stamp = self.get_clock().now().to_msg()

        pose_stamped.pose.position.x = t[0]
        pose_stamped.pose.position.y = t[1]
        pose_stamped.pose.position.z = t[2]

        pose_stamped.pose.orientation.x = q[0]
        pose_stamped.pose.orientation.y = q[1]
        pose_stamped.pose.orientation.z = q[2]
        pose_stamped.pose.orientation.w = q[3]

        self.path_msg.poses.append(pose_stamped)
        self.path_msg.header.stamp = self.get_clock().now().to_msg()

        self.path_pub.publish(self.path_msg)

        self.index += 1

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
