import cv2
import numpy as np
import yaml
import sys
import os
import random
from scipy.spatial.transform import Rotation
import rosbag2_py
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import rclpy
from rclpy.node import Node


# Cargamos la calibracion de las camaras
def load_calib(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return data

class Script:
    def __init__(self,cam_info_l, cam_info_r):
        self.cam_info_l = None
        self.cam_info_r = None
        self.dist_coef_l = None
        self.K_l = None
        self.K_r = None
        self.P_l = None
        self.P_r = None
        self.Q = None
        self.t = None
        self.E = None
        self.R = None
        self.M = None
        # Cargamos las calibraciones
        self.cam_info_l = cam_info_l
        self.cam_info_r = cam_info_r

    def loadImagesFromBag(self, bag_path):

        topic_left="/cam0/image_raw"
        topic_right="/cam1/image_raw"

        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions('', '')
        reader.open(storage_options, converter_options)

        bridge = CvBridge()
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
        
    # Cargamos la informacion de calibracion de las camaras en variables locales
    def loadCameraInfoVariables(self):

        cam_info_l, cam_info_r = self.cam_info_l, self.cam_info_r

        # Obtenemos el tamano de las camaras
        # Como ambas camaras tienen el mismo tamano no importa de cual lo obtengamos
        self.img_size = (cam_info_l['image_width'], cam_info_l['image_height']) 

        # Obtenemos los coeficientes de distorsión
        self.dist_coef_l = np.array(cam_info_l['distortion_coefficients']['data'])
        self.dist_coef_r = np.array(cam_info_r['distortion_coefficients']['data'])

        # Extraemos las matrices de rotacion y traslacion de las camaras
        self.K_l = np.array(cam_info_l['camera_matrix']['data']).reshape(3, 3)
        self.P_l = np.array(cam_info_l['projection_matrix']['data']).reshape(3, 4)
        self.K_r = np.array(cam_info_r['camera_matrix']['data']).reshape(3, 3)
        self.P_r = np.array(cam_info_r['projection_matrix']['data']).reshape(3, 4)

    # Rectificamos 
    def rectify(self, img_l, img_r):

        # Cargamos las varibles de la clase a utilizar
        img_size, dist_coef_l, dist_coef_r, K_l, K_r, P_l, P_r = self.img_size, self.dist_coef_l, self.dist_coef_r, self.K_l, self.K_r, self.P_l, self.P_r

        # Extraemos las partes [R|t] de la proyección
        Rt_l = np.linalg.inv(K_l) @ P_l
        Rt_r = np.linalg.inv(K_r) @ P_r

        # Extraemos R y t de ambas camaras
        R_l = Rt_l[:, :3]
        R_r = Rt_r[:, :3]   
        t_l = Rt_l[:, 3]
        t_r = Rt_r[:, 3]

        # Obtenemos el R y t
        R = R_r @ R_l.T
        t = t_r - R @ t_l

        # Calculamos las matrices de rectificación
        R_l, R_r, P_l, P_r, Q, _, _ = cv2.stereoRectify(
            cameraMatrix1=K_l,
            distCoeffs1=dist_coef_l,
            cameraMatrix2=K_r,
            distCoeffs2=dist_coef_r,
            imageSize=img_size,
            R=R,
            T=t
        )

        # Calculamos los mapas de remapeo para las cámaras izquierda y derecha
        map1_l, map2_l = cv2.initUndistortRectifyMap(K_l, dist_coef_l, R_l, P_l, img_size, cv2.CV_32FC1)
        map1_r, map2_r = cv2.initUndistortRectifyMap(K_r, dist_coef_r, R_r, P_r, img_size, cv2.CV_32FC1)

        # Aplicamos el remapeo a las imágenes
        rectified_left = cv2.remap(img_l, map1_l, map2_l, interpolation=cv2.INTER_LINEAR)
        rectified_right = cv2.remap(img_r, map1_r, map2_r, interpolation=cv2.INTER_LINEAR)

        # Guardamos las variables a utilizar en otros metodos
        self.R, self.t, self.R_l, self.R_r, self.Q = R, t, R_l, R_r, Q

        return rectified_left, rectified_right

    # Obtenemos los key points y los descriptores
    def getFeatures(self, img_l, img_r):

        # Obtenemos los keypoints
        fast = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
        key_pts_l = fast.detect(img_l, None)
        key_pts_r = fast.detect(img_r, None)

        # Obtenemos los descriptores
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        key_pts_l, des_l = brief.compute(img_l, key_pts_l)
        key_pts_r, des_r = brief.compute(img_r, key_pts_r)

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
        pts_4D = cv2.triangulatePoints(self.P_l, self.P_r, pts_l.T, pts_r.T)
        pts_3D = (pts_4D[:3] / pts_4D[3]).T

        return pts_3D

    # Filtramos los matches 
    def filterMatches(self, matches, pts_l, pts_r):

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

    def to_world(self, points, pose, scale=1.0):

        # Aplanamos los puntos y filtramos NaNs
        pts = points.reshape(-1, 3)
        mask = ~np.isnan(pts).any(axis=1)
        pts = pts[mask]

        # Obtenemos traslación y rotación
        t = np.array(pose[1:4])
        r = Rotation.from_quat(pose[4:8]).as_matrix()  # scipy espera [x, y, z, w]

        # Transformamos y aplicamos escala
        points_world = scale * (r @ pts.T).T + t

        return points_world.astype(np.float32)

    # def to_points_cloud2(self, points, frame_id='map'):



    # Computamos el mapa de disparidad
    def computeDisparityMap(self, left_image, right_image):
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

        # Calculamos el mapa de disparidad. El resultado está codificado y necesita ser dividido por 16.0
        # y convertido a float32 para la reproyección 3D.
        disparity_map = stereo.compute(left_image, right_image).astype(np.float32) / 16.0
        
        # Guardamos el mapa de disparidad real (flotante) para la reproyección 3D (Ejercicio H)
        self.disparity = disparity_map 
        
        # Normalizamos la imagen para que sea visible (uint8, 0-255)
        # Esto es lo que se devuelve para mostrar con cv2.imshow (Ejercicio G)
        disparity_visual = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        return disparity_visual

    # Reconstruimos la escena 3D
    def rebuildDense3DScene(self):
        points3D = cv2.reprojectImageTo3D(self.disparity, self.Q).astype(np.float32)
        return points3D

    # Estimamos la pose de las camaras
    def stimatePose(self, pts_l, pts_r):

        # Obtenemos la matriz esencial
        E, mask = cv2.findEssentialMat(pts_l, pts_r, self.M)

        # Obtenemos la transformacion entre la camara izquierda y derecha
        _, R_est, t, mask = cv2.recoverPose(E, pts_l, pts_r, self.K_l)

        # Obtenemos el base line
        f_x = self.P_l[0,0]
        t_x_r = self.P_r[0,3]   
        baseline = -t_x_r / f_x

        # Obtenemos la traslacion escalada
        t_scaled = t * baseline

        # Suponemos la posicion de la camara izquierda se encuetra en el centro de la imagen izquierda
        camera_pose_l = np.array([0.0, 0.0, 0.0])
        camera_pose_r = t_scaled.reshape(3) 

        # Obtenemos la escala de la imagen
        img_width = self.img_size[0]
        scale = 1.0 / f_x * img_width  # factor relativo

        # Pasamos los pose a coordenadas de la camara
        img_size = self.img_size
        camera_pose_l = (int(img_size[0]/2 + camera_pose_l[0]*scale), int(img_size[1] - camera_pose_l[2]*scale))
        camera_pose_r = (int(img_size[0]/2 + camera_pose_r[0]*scale), int(img_size[1] - camera_pose_r[2]*scale))

        return camera_pose_l, camera_pose_r

class ImagePublisher(Node):
    def __init__(self, input_dir, output_dir, save):
        super().__init__('image_publisher')

        # Creamos los Publishers
        # Imagenes
        self.img_pub_l = self.create_publisher(Image, '/Publisher/Left/raw_img', 10)
        self.img_pub_r = self.create_publisher(Image, '/Publisher/Right/raw_img', 10)
        # Imagenes Rectificadas
        self.rect_img_pub_l = self.create_publisher(Image, '/Publisher/Left/rect_img', 10)
        self.rect_img_pub_r = self.create_publisher(Image, '/Publisher/Right/rect_img', 10)
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
        # Escena Rebuildeada
        self.rebuilded_scene_pub = self.create_publisher(PointCloud2, '/Publisher/rebuilded_scene', 10)
        # Poses
        self.poses_pub = self.create_publisher(Image, '/Publisher/poses', 10)
        # Trayectoria
        self.trayectory_pub = self.create_publisher(Path, '/Publisher/trayectory', 10)


        # Generamos los path de los elementos del input
        bag_path = os.path.join(input_dir, "rosbag.db3")
        poses_path = os.path.join(input_dir, "poses.txt")
        calib_path_l = os.path.join(input_dir, "calibration_left.yaml")
        calib_path_r = os.path.join(input_dir, "calibration_right.yaml")

        # Cargamos la informacion del input
        poses = np.loadtxt(poses_path)
        cam_info_l = load_calib(calib_path_l)
        cam_info_r = load_calib(calib_path_r)

        # Definimos el largo y los indices
        len = round(poses.size / 8)
        indices = range(len)
        if save:
            indices = [random.randint(0, len - 1)]

        # Generamos
        script = Script(cam_info_l, cam_info_r)

        # Cargamos la informacion de la camara
        script.loadCameraInfoVariables()

        # Cargamos las imagenes de la rosbag
        imgs_l, imgs_r = script.loadImagesFromBag(bag_path)

        #
        self.trayectory_msg = Path()
        self.trayectory_msg.header.frame_id = "map"

        for i in indices:
            if save or i % 10 == 0:
                print(f"Procesando frame {i+1}/{round(poses.size / 8)}...")

                # Cargamos las imagenes
                img_l = imgs_l[0]
                img_r = imgs_r[1]

                # Guardamos o publicamos las imagenes
                if save:
                    cv2.imwrite(f'{output_dir}/Rectified/Left/rectified_left_{i:04d}.png', rect_img_l)
                    cv2.imwrite(f'{output_dir}/Rectified/Right/rectified_right_{i:04d}.png', rect_img_r)
                else:
                    self.img_pub_l.publish(imgs_l)
                    self.img_pub_r.publish(imgs_r)

                # Ejercicio A

                # Rectificamos las imagenes
                rect_img_l, rect_img_r = script.rectify(img_l, img_r)

                # Guardamos o publicamos las imagenes rectificadas
                if save:
                    cv2.imwrite(f'{output_dir}/Rectified/Left/rectified_left_{i:04d}.png', rect_img_l)
                    cv2.imwrite(f'{output_dir}/Rectified/Right/rectified_right_{i:04d}.png', rect_img_r)
                else:
                    self.rect_img_pub_l.publish(rect_img_l)
                    self.rect_img_pub_r.publish(rect_img_r)
                
                # Ejercicio B

                # Obtenemos los features 
                key_pts_l, key_pts_r = script.getFeatures(rect_img_l, rect_img_r)

                # Imprimimos los resultados
                img_key_pts_l = cv2.drawKeypoints(rect_img_l, key_pts_l, None, color=(0,255,0))
                img_key_pts_r = cv2.drawKeypoints(rect_img_r, key_pts_r, None, color=(0,255,0))

                # Guardamos o publicamos los keypoints
                if save:
                    cv2.imwrite(f'{output_dir}/Keypoints/Left/keypoints_left_{i:04d}.png', img_key_pts_l)
                    cv2.imwrite(f'{output_dir}/Keypoints/Right/keypoints_right_{i:04d}.png', img_key_pts_r)
                else:
                    self.keypoints_img_pub_l.publish(img_key_pts_l)
                    self.keypoints_img_pub_r.publish(img_key_pts_r)

                # Ejercicio C

                # Obtenemos todos los matches
                all_matches = script.getMatches()

                # Filtramos los matches buenos con distancia mayor a 30
                good_matches = [m for m in all_matches if m.distance < 30]

                # Imprimimos los resultados
                img_all_matches = cv2.drawMatches(rect_img_l, key_pts_l, img_r, key_pts_r, all_matches, None)
                img_good_matches = cv2.drawMatches(rect_img_r, key_pts_l, img_r, key_pts_r, good_matches, None)

                # Guardamos o publicamos todos los matches y los matches buenos
                if save:
                    cv2.imwrite(f'{output_dir}/Matches/all_matches_{i:04d}.png', img_all_matches)
                    cv2.imwrite(f'{output_dir}/Matches/good_matches_{i:04d}.png', img_good_matches)
                else:
                    self.all_matches_pub.publish(img_all_matches)
                    self.good_matches_pub.publish(img_good_matches)

                # Ejercicio D

                # Obtenems los puntos de los matches buenos
                good_pts_l, good_pts_r = script.getMatchesPoints(good_matches)

                # Triangulamos los puntos
                good_pts_3D = script.triangulate(good_pts_l, good_pts_r)

                # Ejercicio E

                # Filtramos los matches espureos
                filtered_matches = script.filterMatches(good_matches, good_pts_l, good_pts_r)

                # Generamos la imagen con los matches filtrados
                img_filtered_matches = cv2.drawMatches(rect_img_l, key_pts_l, rect_img_r, key_pts_r, filtered_matches, None)

                # Guardamos o publicamos los matches filtrados
                if save:
                    cv2.imwrite(f'{output_dir}/Matches/filtered_matches_{i:04d}.png', img_filtered_matches)
                else:
                    self.filtered_matches_pub.publish(img_filtered_matches)

                # Transformamos los puntos a la imagen derecha utilizando la matriz M
                filtered_pts_l, filtered_pts_r = script.getMatchesPoints(filtered_matches)
                filtered_pts_transformed_r = script.transformPoints(filtered_pts_l)

                # Dibujamos círculos verdes para los puntos transformados
                img_transformed = rect_img_r.copy()
                for pt in filtered_pts_transformed_r:
                    x, y = pt[0]
                    cv2.circle(img_transformed, (int(x), int(y)), 4, (0, 255, 0), -1)

                # Generamos o publicamos la imagen con los puntos transformados
                if save:
                    cv2.imwrite(f'{output_dir}/Transformed/transformed_points_r{i:04d}.png', img_transformed)
                else:
                    self.transformed_points_pub.publish(img_transformed)

                # Ejercicio F

                # Pasamos los puntos buenos al sistema de coordenadas global
                good_pts_3d_world = script.to_world(good_pts_3D, poses[i])

                # Guardamos o publicamos los puntos buenos 3d
                if save:
                    np.save(f"{output_dir}/3DPoints/good_points3D_{i:04d}.npy", good_pts_3d_world)
                else:
                    self.good_points_3D_pub.publish(good_pts_3d_world)

                # Triangulamos los puntos filtrados
                filtered_pts_3D = script.triangulate(filtered_pts_l, filtered_pts_r)

                # Pasamos los puntos filtrados al sistema de coordenadas global
                filtered_pts_3D_world = script.to_world(filtered_pts_3D, poses[i])

                # Guardamos o publicamos los puntos filtrados 3d
                if save:
                    np.save(f"{output_dir}/Filtered3DPoints/filtered_points3D_{i:04d}.npy", filtered_pts_3D_world)
                else:
                    self.filtered_points_3D_pub.publish(filtered_pts_3D_world)

                # Ejercicio G

                # Obtenemos el mapa de disparidad
                disparity_map = script.computeDisparityMap(rect_img_l, rect_img_r)

                # Guardamos o publicamos el mapa de disparidad
                if save:
                    cv2.imwrite(f'{output_dir}/Disparities/disparity_map_{i:04d}.png', disparity_map)
                else:
                    self.disparity_pub.publish(disparity_map)

                # Ejercicio H

                # Rebuildeamos la escena 3D
                rebuilded_pts_3D = script.rebuildDense3DScene()

                # Ejercicio I
                rebuilded_pts_3D_world = script.to_world(rebuilded_pts_3D, poses[i])

                # Guardamos o publicamos la escena rebuildeada
                if save:
                    np.save(f'{output_dir}/RebiuldedScenes/rebuilded_pts_3D_world_{i:04d}.npy', rebuilded_pts_3D_world)
                else:
                    self.rebuilded_scene_pub.publish(rebuilded_pts_3D_world)

                # Ejercicio J

                # Subindice i

                # Agregamos a la imagen izquierda los poses de las camaras
                img_poses = rect_img_r.copy()

                camera_pose_l, camera_pose_r = script.stimatePose(filtered_pts_l, filtered_pts_r)

                # Dibujamos los centros de las camaras
                cv2.circle(img_poses, camera_pose_l, 8, (0,0,255), -1)
                cv2.circle(img_poses, camera_pose_r, 8, (255,0,0), -1)

                # Dibujamos una baseline
                cv2.line(img_poses, camera_pose_l, camera_pose_r, (0,0,0), 2, cv2.LINE_AA)

                # Guardamos o publicamos las poses de la camara
                if save:
                    cv2.imwrite(f'{output_dir}/CamerasPoses/cameras_pose_{i:04d}.png', img_poses)
                else:
                    self.poses_pub.publish(img_poses)

                # Subindice ii

                if save:
                    t = poses[i][1:4]
                    q = poses[i][4:]

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

                    self.trayectory_msg.poses.append(pose_stamped)
                    self.trayectory_msg.header.stamp = self.get_clock().now().to_msg()
                    
                    self.trayectory_pub.publish(self.trayectory_msg)

def main():

    if len(sys.argv) < 4:
        print("Uso: python3 script.py <input_dir> <output_dir> [--save True]")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    save = len(sys.argv) > 3 and sys.argv[3].lower() == "true"

    if save:
        os.makedirs(os.path.join(output_dir, "Rectified/Left"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "Rectified/Right"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "Keypoints/Left"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "Keypoints/Right"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "Matches"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "Transformed"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "3DPoints"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "Filtered3DPoints"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "Disparities"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "RebiuldedScenes"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "CamerasPoses"), exist_ok=True)
    
    rclpy.init()
    node = ImagePublisher(input_dir, output_dir, save)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()