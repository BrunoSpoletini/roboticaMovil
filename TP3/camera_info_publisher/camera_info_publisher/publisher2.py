import os
import rosbag2_py
import cv2
import yaml
import sys
import time
import numpy as np
from sensor_msgs.msg import Image, PointCloud2, PointField
from nav_msgs.msg import Path
from cv_bridge import CvBridge
import rclpy
from rclpy.serialization import deserialize_message
from message_filters import Subscriber, ApproximateTimeSynchronizer
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from scipy.spatial.transform import Rotation
from visualization_msgs.msg import Marker
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

input_dir = None
output_dir = None

bridge = None

poses = None

index = 0
poses_len = None

cam_info_l = None
cam_info_r = None
dist_coef_l = None
dist_coef_r = None
img_size = None

K_l = None
K_r = None
P_l = None
P_r = None
P_rect_l = None
P_rect_r = None
R_rect_l = None
Q = None
t = None
R = None
M = None

T_imu_cam_l = None
T_imu_cam_r = None

T_cam_imu_l = None
T_cam_imu_r = None

# Cargamos la calibracion de las camaras
def load_calib(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return data

# Cargamos la informacion de calibracion de las camaras en variables locales
def loadCameraInfoVariables():

    global img_size, dist_coef_l, dist_coef_r, K_l, P_l, K_r, P_r

    # Obtenemos el tamano de las camaras
    # Como ambas camaras tienen el mismo tamano no importa de cual lo obtengamos
    img_size = (cam_info_l['image_width'], cam_info_l['image_height']) 

    # Obtenemos los coeficientes de distorsión
    dist_coef_l = np.array(cam_info_l['distortion_coefficients']['data'])
    dist_coef_r = np.array(cam_info_r['distortion_coefficients']['data'])

    # Extraemos las matrices de rotacion y traslacion de las camaras
    K_l = np.array(cam_info_l['camera_matrix']['data']).reshape(3, 3)
    P_l = np.array(cam_info_l['projection_matrix']['data']).reshape(3, 4)
    K_r = np.array(cam_info_r['camera_matrix']['data']).reshape(3, 3)
    P_r = np.array(cam_info_r['projection_matrix']['data']).reshape(3, 4)

# 
def numpy_to_ros_image(img_array):
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        return bridge.cv2_to_imgmsg(img_array, encoding='bgr8')
    elif len(img_array.shape) == 2 or (len(img_array.shape) == 3 and img_array.shape[2] == 1):
        return bridge.cv2_to_imgmsg(img_array, encoding='mono8')
    
# Pasa los puntos al sistema de coordenadas del mundo
def to_world(points, pose):
        
        pts = points.reshape(-1, 3)
        pts = pts[~np.isnan(pts).any(axis=1)]
        
        if pts.shape[0] == 0:
            return np.array([]).reshape(0, 3).astype(np.float32)

        # Pasamos los puntos desde la camara rectifica a la camara original
        pts_cam_l = (R_rect_l.T @ pts.T).T

        # Obtenemos la pose del IMU en el mundo
        t_imu_world = np.array(pose[1:4])
        q_imu_world = pose[4:8]
        R_imu_world = Rotation.from_quat(q_imu_world).as_matrix()

        # Obtenemos la rotacion y la traslacion de la camara al IMU
        R_imu_cam_l = T_imu_cam_l[:3, :3]
        t_imu_cam_l = T_imu_cam_l[:3, 3].reshape(1, 3)

        # Transformamos de la camara al IMU
        pts_imu = (R_imu_cam_l @ pts_cam_l.T).T + t_imu_cam_l
        
        # Transformamos del IMU al mundo
        pts_world = (R_imu_world @ pts_imu.T).T + t_imu_world

        return pts_world.astype(np.float32)

def load_extrinsics(yaml_path):

    global T_imu_cam_l, T_imu_cam_r, T_cam_imu_l, T_cam_imu_r

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # Obtenemos las transformaciones de las camaras a la IMU
    T_imu_cam_l = np.array(data["cam0"]["T_imu_cam"])
    T_imu_cam_r = np.array(data["cam1"]["T_imu_cam"])

    # Obtenemos las transformaciones de IMU a las camara
    T_cam_imu_l = np.linalg.inv(T_imu_cam_l)
    T_cam_imu_r = np.linalg.inv(T_imu_cam_r)

def points_to_pointcloud2(points, colors=None, frame_id='map'):
        header = Header()
        header.stamp = rclpy.time.Time().to_msg()
        header.frame_id = frame_id

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]

        if colors is not None:
            fields.append(PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1))
            point_step = 16
        else:
            point_step = 12

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = points.shape[0]
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = point_step
        msg.row_step = msg.point_step * points.shape[0]
        msg.is_dense = False  # Set to False because we may have filtered some points

        if colors is not None:
            r = colors[:, 0].astype(np.uint32)
            g = colors[:, 1].astype(np.uint32)
            b = colors[:, 2].astype(np.uint32)
            rgb_values = (r << 16) | (g << 8) | b
            
            # Creamos un array estructurado para combinar xyz y rgb correctamente
            cloud_data = np.zeros((points.shape[0],), dtype=[
                ('x', np.float32),
                ('y', np.float32),
                ('z', np.float32),
                ('rgb', np.uint32)
            ])
            cloud_data['x'] = points[:, 0]
            cloud_data['y'] = points[:, 1]
            cloud_data['z'] = points[:, 2]
            cloud_data['rgb'] = rgb_values
            
            data = cloud_data.tobytes()
        else:
            data = points.astype(np.float32).tobytes()

        msg.data = data

        return msg

# Publicador de Imagenes
class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')

        global poses_len

        # Genermos el path de la rosbag
        bag_path = os.path.join(input_dir, "rosbag.db3")

        # Obtenemos las imagenes
        self.imgs_l, self.imgs_r = self.loadImagesFromBag(bag_path)

        # Generamos los publicadores
        self.img_pub_l = self.create_publisher(Image, '/Publisher/Left/raw_img', 10)
        self.img_pub_r = self.create_publisher(Image, '/Publisher/Right/raw_img', 10)

        # Calculamos la cantidad de poses
        poses_len = round(poses.size / 8)

        self.timer = self.create_timer(0.1, self.publish_next_img)

    def publish_next_img(self):

        global index

        if index < poses_len:    
            index = index + 1
        else:
            index = 0

        print(f"Procesando imagen {index+1}/{len}...")

        # Cargamos las imagenes
        img_l = self.imgs_l[index]
        img_r = self.imgs_r[index]

        # Publicamos las imagenes
        self.img_pub_l.publish(numpy_to_ros_image(img_l))
        self.img_pub_r.publish(numpy_to_ros_image(img_r))


    def loadImagesFromBag(self, bag_path):

        topic_left="/cam0/image_raw"
        topic_right="/cam1/image_raw"

        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions('', '')
        reader.open(storage_options, converter_options)

        left_images = []
        right_images = []

        while reader.has_next():
            (topic, data, t) = reader.read_next()
            if topic == topic_left:
                msg = deserialize_message(data, Image)
                img_l = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                left_images.append(img_l)
            elif topic == topic_right:
                msg = deserialize_message(data, Image)
                img_r = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                right_images.append(img_r)
        
        return left_images, right_images
    
class ImageRectifier(Node):
    def __init__(self):
        super().__init__('image_rectifier')

        global P_rect_l, P_rect_r, R_rect_l, Q, R, t

        # Extraemos las partes [R|t] de la proyección
        Rt_l = np.linalg.inv(K_l) @ P_l
        Rt_r = np.linalg.inv(K_r) @ P_r

        # Extraemos R y t de ambas camaras
        R_l_orig = Rt_l[:, :3]
        R_r_orig = Rt_r[:, :3]   
        t_l = Rt_l[:, 3]
        t_r = Rt_r[:, 3]

        # Obtenemos el R y t relativos entre cámaras
        R = R_r_orig @ R_l_orig.T
        t = t_r - R @ t_l

        # Calculamos las matrices de rectificación
        R_rect_l, R_rect_r, P_rect_l, P_rect_r, Q, _, _ = cv2.stereoRectify(
            cameraMatrix1=K_l,
            distCoeffs1=dist_coef_l,
            cameraMatrix2=K_r,
            distCoeffs2=dist_coef_r,
            imageSize=img_size,
            R=R,
            T=t
        )

        # Calculamos los mapas de remapeo para las cámaras izquierda y derecha
        self.map1_l, self.map2_l = cv2.initUndistortRectifyMap(K_l, dist_coef_l, R_rect_l, P_rect_l, img_size, cv2.CV_32FC1)
        self.map1_r, self.map2_r = cv2.initUndistortRectifyMap(K_r, dist_coef_r, R_rect_r, P_rect_r, img_size, cv2.CV_32FC1)

        self.rect_img_pub_l = self.create_publisher(Image, '/Publisher/Left/rect_img', 10)
        self.rect_img_pub_r = self.create_publisher(Image, '/Publisher/Right/rect_img', 10)

        self.sub_img_l = Subscriber(self, Image, '/Publisher/Left/raw_img')
        self.sub_img_r = Subscriber(self, Image, '/Publisher/Right/raw_img')

        self.sync = ApproximateTimeSynchronizer(
            [self.sub_img_l, self.sub_img_r],
            queue_size=10,
            slop=0.1
        )

        self.sync.registerCallback(self.callback_rectify_image)

    def callback_rectify_image(self, img_l, img_r):

        img_cv_l = bridge.imgmsg_to_cv2(img_l, desired_encoding='bgr8')
        img_cv_r = bridge.imgmsg_to_cv2(img_r, desired_encoding='bgr8')

        # Aplicamos el remapeo a las imágenes
        rect_img_l = cv2.remap(img_cv_l, self.map1_l, self.map2_l, interpolation=cv2.INTER_LINEAR)
        rect_img_r = cv2.remap(img_cv_r, self.map1_r, self.map2_r, interpolation=cv2.INTER_LINEAR)

        # Publicamos las imagenes rectificadas
        self.rect_img_pub_l.publish(numpy_to_ros_image(rect_img_l))
        self.rect_img_pub_r.publish(numpy_to_ros_image(rect_img_r))


class MatchesFinder(Node):
    def __init__(self):
        super().__init__('matches_finder')

        print(f"Obteniendo features {index+1}/{len}...")

        # Keypoints
        self.keypoints_img_pub_l = self.create_publisher(Image, '/Publisher/Left/keypoints_img', 10)
        self.keypoints_img_pub_r = self.create_publisher(Image, '/Publisher/Right/keypoints_img', 10)
        # Matches
        self.all_matches_pub = self.create_publisher(Image, '/Publisher/all_matches', 10)
        self.good_matches_pub = self.create_publisher(Image, '/Publisher/good_matches', 10)
        self.filtered_matches_pub = self.create_publisher(Image, '/Publisher/filtered_matches', 10)
        # Transformed Points
        self.transformed_points_pub = self.create_publisher(Image, '/Publisher/transformed_points', 10)
        # Mapa de disparidad
        self.disparity_pub = self.create_publisher(Image, '/Publisher/disparity_map', 10)
        # Nube de puntos
        self.good_points_3D_pub = self.create_publisher(PointCloud2, '/Publisher/good_points', 10)
        self.filtered_points_3D_pub = self.create_publisher(PointCloud2, '/Publisher/filtered_points', 10)

        self.sub_rect_img_l = Subscriber(self, Image, '/Publisher/Left/rect_img')
        self.sub_rect_img_r = Subscriber(self, Image, '/Publisher/Right/rect_img')

        self.sync = ApproximateTimeSynchronizer(
            [self.sub_rect_img_l, self.sub_rect_img_r],
            queue_size=10,
            slop=0.1
        )

        self.sync.registerCallback(self.callback_find_matches)

    def callback_find_matches(self, img_l, img_r):

        print(f"Obteniendo features {index+1}/{len}...")

        img_cv_l = bridge.imgmsg_to_cv2(img_l, desired_encoding='bgr8')
        img_cv_r = bridge.imgmsg_to_cv2(img_r, desired_encoding='bgr8')

        # Obtenemos los features 
        key_pts_l, key_pts_r = self.getFeatures(img_cv_l, img_cv_r)

        # Imprimimos los resultados
        img_key_pts_l = cv2.drawKeypoints(img_cv_l, key_pts_l, None, color=(0,255,0))
        img_key_pts_r = cv2.drawKeypoints(img_cv_r, key_pts_r, None, color=(0,255,0))

        # Publicamos los keypoints
        self.keypoints_img_pub_l.publish(numpy_to_ros_image(img_key_pts_l))
        self.keypoints_img_pub_r.publish(numpy_to_ros_image(img_key_pts_r))

        print(f"Obteniendo matches {index+1}/{len}...")

        # Obtenemos todos los matches
        all_matches = self.getMatches()

        # Filtramos los matches buenos con distancia mayor a 30
        good_matches = [m for m in all_matches if m.distance < 30]

        # Imprimimos los resultados
        img_all_matches = cv2.drawMatches(img_cv_l, key_pts_l, img_cv_r, key_pts_r, all_matches, None)
        img_good_matches = cv2.drawMatches(img_cv_l, key_pts_l, img_cv_r, key_pts_r, good_matches, None)

        # Publicamos todos los matches y los matches buenos
        self.all_matches_pub.publish(numpy_to_ros_image(img_all_matches))
        self.good_matches_pub.publish(numpy_to_ros_image(img_good_matches))

        print(f"Triangulando puntos {index+1}/{len}...")

        # Obtenems los puntos de los matches buenos
        good_pts_l, good_pts_r = self.getMatchesPoints(good_matches)

        # Triangulamos los puntos
        good_pts_3D = self.triangulate(good_pts_l, good_pts_r)

        print(f"Filtrando matches {index+1}/{len}...")

        # Filtramos los matches espureos
        filtered_matches = self.filterMatches(good_matches, good_pts_l, good_pts_r)

        # Generamos la imagen con los matches filtrados
        img_filtered_matches = cv2.drawMatches(img_cv_l, key_pts_l, img_cv_r, key_pts_r, filtered_matches, None)

        # Publicamos los matches filtrados
        self.filtered_matches_pub.publish(numpy_to_ros_image(img_filtered_matches))

        print(f"Transformando puntos {index+1}/{len}...")

        # Transformamos los puntos a la imagen derecha utilizando la matriz M
        filtered_pts_l, filtered_pts_r = self.getMatchesPoints(filtered_matches)
        filtered_pts_transformed_r = self.transformPoints(filtered_pts_l)

        # Dibujamos círculos verdes para los puntos transformados
        img_transformed = img_cv_r.copy()
        for pt in filtered_pts_transformed_r:
            x, y = pt[0]
            cv2.circle(img_transformed, (int(x), int(y)), 4, (0, 255, 0), -1)

        # Publicamos la imagen con los puntos transformados
        self.transformed_points_pub.publish(numpy_to_ros_image(img_transformed))

        print(f"Generando puntos 3D {index+1}/{len}...")

        # Pasamos los puntos buenos al sistema de coordenadas global
        good_pts_3d_world = to_world(good_pts_3D, poses[index])

        # Publicamos los puntos buenos 3d
        good_points_cloud = points_to_pointcloud2(good_pts_3d_world)
        self.good_points_3D_pub.publish(good_points_cloud)

        # Triangulamos los puntos filtrados
        filtered_pts_3D = self.triangulate(filtered_pts_l, filtered_pts_r)

        # Pasamos los puntos filtrados al sistema de coordenadas global
        filtered_pts_3D_world = to_world(filtered_pts_3D, poses[index])

        # Publicamos los puntos filtrados 3d
        filtered_points_cloud = points_to_pointcloud2(filtered_pts_3D_world)
        self.filtered_points_3D_pub.publish(filtered_points_cloud)

    # Obtenemos los key points y los descriptores
    def getFeatures(self, img_l, img_r):

        # Obtenemos los keypoints
        fast = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
        key_pts_l = fast.detect(img_l, None)
        key_pts_r = fast.detect(img_r, None)

        # Obtenemos los descriptores
        orb = cv2.ORB_create(nfeatures=1000)
        key_pts_l, des_l = orb.detectAndCompute(img_l, None)
        key_pts_r, des_r = orb.detectAndCompute(img_r, None)

        # Guardamos las variables a utilizar en otros metodos
        self.key_pts_l, self.key_pts_r, self.des_l, self.des_r = key_pts_l, key_pts_r, des_l, des_r

        return key_pts_l, key_pts_r
    
    # Obtenemos los matches buenos
    def getMatches(self):

        # Cargamos las varibles de la clase a utilizar
        des_l, des_r = self.des_l, self.des_r

        # Generamos el Matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Encontramos los matches
        matches = bf.match(des_l, des_r)

        return matches
    
    # Ordenamos y filtramos los puntos por match
    def getMatchesPoints(self, matches):

        pts_l = np.float32([self.key_pts_l[m.queryIdx].pt for m in matches])
        pts_r = np.float32([self.key_pts_r[m.trainIdx].pt for m in matches])

        return pts_l, pts_r
    
    # Triangulamos los key points
    def triangulate(self, pts_l, pts_r):
        
        # Realizamos la triangulacion
        pts_4D = cv2.triangulatePoints(P_rect_l, P_rect_r, pts_l.T, pts_r.T)
        pts_3D = (pts_4D[:3] / pts_4D[3]).T

        return pts_3D

    # Filtramos los matches 
    def filterMatches(self, matches, pts_l, pts_r):

        global M

        # obtenemos la matriz homografica y la mascara
        M, mask = cv2.findHomography(pts_l, pts_r, cv2.RANSAC, 5.0)

        # Filtramos los matches
        filtered_matches = [m for i, m in enumerate(matches) if mask[i]]

        # Guardamos las variables a utilizar en otros metodos
        self.M = M

        return filtered_matches
    
    def transformPoints(self, pts_l):

        pts_l = np.asarray(pts_l, dtype=np.float32)
        if pts_l.ndim == 2:
            pts_l = pts_l.reshape(-1, 1, 2)

        pts_transformed_r = cv2.perspectiveTransform(pts_l, self.M)

        return pts_transformed_r

class SceneRebuilder(Node):
    def __init__(self):
        super().__init__('scene_rebuilder')

        self.disparity_map_pub = self.create_publisher(Image, '/Publisher/dispariry_map', 10)
        self.rebuilded_scene_pub = self.create_publisher(PointCloud2, '/Publisher/rebuilded_scene', 10)

        self.sub_rect_img_l = Subscriber(self, Image, '/Publisher/Left/rect_img')
        self.sub_rect_img_r = Subscriber(self, Image, '/Publisher/Right/rect_img')

        self.sync = ApproximateTimeSynchronizer(
            [self.sub_rect_img_l, self.sub_rect_img_r],
            queue_size=10,
            slop=0.1
        )

        self.sync.registerCallback(self.callback_rebuild_scene)

    def callback_rebuild_scene(self, img_l, img_r):

        img_cv_l = bridge.imgmsg_to_cv2(img_l, desired_encoding='bgr8')
        img_cv_r = bridge.imgmsg_to_cv2(img_r, desired_encoding='bgr8')

        print(f"Computando disparidad {index+1}/{len}...")

        # Obtenemos el mapa de disparidad
        disparity_map = self.computeDisparityMap(img_cv_l, img_cv_r)

        # Publicamos el mapa de disparidad
        self.disparity_map_pub.publish(numpy_to_ros_image(disparity_map))

        print(f"Rebuildeando escena 3D {index+1}/{len}...")

        # Rebuildeamos la escena 3D
        rebuilded_pts_3D = cv2.reprojectImageTo3D(disparity_map, Q).astype(np.float32)

        # Coloreamos los puntos y transformamos al mundo en una sola operación
        colored_rebuilded_pts_3D = self.color_3d_points(rebuilded_pts_3D, img_cv_l, poses[index])
        
        # Publicamos la escena rebuildeada
        self.rebuilded_scene_pub.publish(colored_rebuilded_pts_3D)        

    # Computamos el mapa de disparidad
    def computeDisparityMap(self, left_image, right_image):
        # Convertimos a escala de grises si es necesario
        if len(left_image.shape) == 3:
            left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_image
            
        if len(right_image.shape) == 3:
            right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        else:
            right_gray = right_image
        
        # Parámetros para StereoSGBM (pueden requerir ajuste fino para tu dataset)
        min_disp = 0
        
        # numDisparities: Debe ser divisible por 16. Define el rango de búsqueda.
        # Un valor común es 64, 128 o 160.
        num_disp = 128
        
        # blockSize: Debe ser impar (3, 5, 7, etc.). Define el tamaño de la ventana de coincidencias.
        block_size = 5 

        # Creamos el objeto StereoSGBM
        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            
            # P1 y P2 controlan la suavidad. Se calculan basados en blockSize.
            P1=8 * block_size**2,
            P2=32 * block_size**2,
            
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )

        # Calculamos el mapa de disparidad usando las imágenes en escala de grises
        # El resultado está codificado y necesita ser dividido por 16.0
        # y convertido a float32 para la reproyección 3D.
        disparity_map = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
        
        # Guardamos el mapa de disparidad real (flotante) para la reproyección 3D (Ejercicio H)
        self.disparity = disparity_map 
        
        # Normalizamos la imagen para que sea visible (uint8, 0-255)
        # Esto es lo que se devuelve para mostrar con cv2.imshow (Ejercicio G)
        disparity_visual = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        return disparity_visual
    
    def color_3d_points(self, points_3d, rect_img, pose=None):
        
        # Reshape points_3d para tener la forma correcta
        points_reshaped = points_3d.reshape(-1, 3)
        colors_reshaped = rect_img.reshape(-1, 3)
        
        # Filtramos puntos inválidos (infinitos o con Z <= 0, o dist > 20)
        valid_mask = np.isfinite(points_reshaped).all(axis=1) & (points_reshaped[:, 2] > 0) & (np.linalg.norm(points_reshaped, axis=1) < 20)
        
        valid_points = points_reshaped[valid_mask]
        valid_colors = colors_reshaped[valid_mask]
        
        # Tomamos solo 1 de cada 5 puntos para reducir la densidad
        valid_points = valid_points[::15]
        valid_colors = valid_colors[::15]
        
        # Si se proporciona una pose, transformamos al mundo
        if pose is not None:
            valid_points = to_world(valid_points, pose)
        
        # Creamos el PointCloud2 con colores (BGR -> RGB)
        colored_point_cloud = points_to_pointcloud2(valid_points, valid_colors[:, ::-1], frame_id='map')
        
        return colored_point_cloud

class TrayectoryPublisher(Node):
    def __init__(self):
        super().__init__('trayectory_publisher')

        self.cam_markers_pub = self.create_publisher(Marker, '/Publisher/camera_markers', 10)
        self.trayectory_pub = self.create_publisher(Path, '/Publisher/trayectory', 10)

        self.trayectory_msg = Path()
        self.trayectory_msg.header.frame_id = 'map'

        self.sub_rect_img_l = Subscriber(self, Image, '/Publisher/Left/rect_img')
        self.sub_rect_img_r = Subscriber(self, Image, '/Publisher/Right/rect_img')

        self.sync = ApproximateTimeSynchronizer(
            [self.sub_rect_img_l, self.sub_rect_img_r],
            queue_size=10,
            slop=0.1
        )

        self.sync.registerCallback(self.callback_publish_trayectory)

    def callback_publish_trayectory(self, img_l, img_r):

        print(f"Generando trayectoria {index+1}/{len}...")

        t = poses[index][1:4]
        q = poses[index][4:]

        # Pose IMU 
        T_imu_world = np.eye(4)
        T_imu_world[:3, :3] = Rotation.from_quat(q).as_matrix()
        T_imu_world[:3, 3] = t

        # Obtenemos la pose de las camaras en el mundo
        T_cam_world_l = T_imu_world @ T_imu_cam_l
        T_cam_world_r = T_imu_world @ T_imu_cam_r

        # Extraemos traslación y rotación de las camaras
        t_cam_l = T_cam_world_l[:3, 3]
        Q_cam_l = Rotation.from_matrix(T_cam_world_l[:3, :3]).as_quat()
        t_cam_r = T_cam_world_r[:3, 3]
        Q_cam_r = Rotation.from_matrix(T_cam_world_r[:3, :3]).as_quat()

        # Generamos los puntos de las camaras

        marker_l = Marker()
        marker_l.header.frame_id = "map"
        marker_l.header.stamp = self.get_clock().now().to_msg()
        marker_l.ns = "cameras"
        marker_l.id = 0
        marker_l.type = Marker.SPHERE
        marker_l.action = Marker.ADD
        marker_l.pose.position.x = t_cam_l[0]
        marker_l.pose.position.y = t_cam_l[1]
        marker_l.pose.position.z = t_cam_l[2]
        marker_l.scale.x = 0.05
        marker_l.scale.y = 0.05
        marker_l.scale.z = 0.05
        marker_l.color.a = 1.0
        marker_l.color.r = 0.0
        marker_l.color.g = 0.0
        marker_l.color.b = 1.0  # azul

        # --- Crear marcador para cámara derecha (rojo) ---
        marker_r = Marker()
        marker_r.header.frame_id = "map"
        marker_r.header.stamp = self.get_clock().now().to_msg()
        marker_r.ns = "cameras"
        marker_r.id = 1
        marker_r.type = Marker.SPHERE
        marker_r.action = Marker.ADD
        marker_r.pose.position.x = t_cam_r[0]
        marker_r.pose.position.y = t_cam_r[1]
        marker_r.pose.position.z = t_cam_r[2]
        marker_r.scale.x = 0.05
        marker_r.scale.y = 0.05
        marker_r.scale.z = 0.05
        marker_r.color.a = 1.0
        marker_r.color.r = 1.0
        marker_r.color.g = 0.0
        marker_r.color.b = 0.0  # rojo

        self.cam_markers_pub.publish(marker_l)
        self.cam_markers_pub.publish(marker_r)

        # Generamos la trayectoria de la camara izquierda
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "map"
        pose_stamped.header.stamp = self.get_clock().now().to_msg()

        pose_stamped.pose.position.x = t_cam_l[0]
        pose_stamped.pose.position.y = t_cam_l[1]
        pose_stamped.pose.position.z = t_cam_l[2]

        pose_stamped.pose.orientation.x = Q_cam_l[0]
        pose_stamped.pose.orientation.y = Q_cam_l[1]
        pose_stamped.pose.orientation.z = Q_cam_l[2]
        pose_stamped.pose.orientation.w = Q_cam_l[3]

        self.trayectory_msg.poses.append(pose_stamped)
        self.trayectory_msg.header.stamp = self.get_clock().now().to_msg()

        # Publicamos la trayectoria de la camara izquierda
        self.trayectory_pub.publish(self.trayectory_msg)

def main():

    global input_dir, output_dir, poses, cam_info_l, cam_info_r, bridge

    if len(sys.argv) < 3:
        print("Uso: python3 script.py <input_dir> <output_dir> [--save True]")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # Generamos los path de los elementos del input
    poses_path = os.path.join(input_dir, "poses.txt")
    calib_path_l = os.path.join(input_dir, "calibration_left.yaml")
    calib_path_r = os.path.join(input_dir, "calibration_right.yaml")
    kalib_imu_cam = os.path.join(input_dir, "kalibr_imucam_chain.yaml")

    # Cargamos la informacion del input
    poses = np.loadtxt(poses_path)
    cam_info_l = load_calib(calib_path_l)
    cam_info_r = load_calib(calib_path_r)
    load_extrinsics(kalib_imu_cam)
    loadCameraInfoVariables()

    bridge = CvBridge()
    
    rclpy.init()

    trayectoryPublisherNode = TrayectoryPublisher()
    sceneRebuilderNode = SceneRebuilder()
    matchesFinderNode = MatchesFinder()
    imageRectifierNode = ImageRectifier()
    imagePublisherNode = ImagePublisher()

    executor = MultiThreadedExecutor()
    executor.add_node(trayectoryPublisherNode)
    executor.add_node(sceneRebuilderNode)
    executor.add_node(matchesFinderNode)
    executor.add_node(imageRectifierNode)
    executor.add_node(imagePublisherNode)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        imageRectifierNode.destroy_node()
        matchesFinderNode.destroy_node()
        sceneRebuilderNode.destroy_node()
        trayectoryPublisherNode.destroy_node()
        imagePublisherNode.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()