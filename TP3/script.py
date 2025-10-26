import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

global cam_info_l
global cam_info_r

output_dir = './'

camera_path_l = './left.png'
camera_path_r = './right.png'

calibration_path_l = './calibrationData/left.yaml'
calibration_path_r = './calibrationData/right.yaml'

# Cargamos la calibracion de las camaras
def load_calib(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return data

class Script:
    def __init__(self):
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


    # Cargamos las imágenes izquierda y derecha
    def loadImages(self, camera_path_l, camera_path_r) :
        img_l = cv2.imread(camera_path_l, cv2.IMREAD_GRAYSCALE)
        img_r = cv2.imread(camera_path_r, cv2.IMREAD_GRAYSCALE)

        return img_l, img_r
        
    # Cargamos la informacion de calibracion de las camaras en variables locales
    def loadCameraInfoVariables(self):

        cam_info_l, cam_info_r = self.cam_info_l, self.cam_info_r

        # Cargamos las calibraciones
        cam_info_l = load_calib(calibration_path_l)
        cam_info_r = load_calib(calibration_path_r)

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

    def to_world(pts, pose):

        t = np.array(pose[:3])
        r = Rotation.from_quat(pose[3:]).as_matrix()
        points_world = (r @ pts.T).T + t
        return points_world

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

def main(args=None):

    script = Script()

    # Ejercicio A

    img_l, img_r = script.loadImages(camera_path_l, camera_path_r)

    script.loadCameraInfoVariables()

    rectified_img_l, rectified_img_r = script.rectify(img_l, img_r)

    # Guardamos las imagenes rectificadas
    cv2.imshow('Rectified Left', rectified_img_l)
    cv2.imshow('Rectified Right', rectified_img_r)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(f'{output_dir}/rectified_left.png', rectified_img_l)
    cv2.imwrite(f'{output_dir}/rectified_right.png', rectified_img_r)

    # Ejercicio B

    key_pts_l, key_pts_r = script.getFeatures(rectified_img_l, rectified_img_r)

    # Imprimimos los resultados
    img_key_pts_l = cv2.drawKeypoints(rectified_img_l, key_pts_l, None, color=(0,255,0))
    img_key_pts_r = cv2.drawKeypoints(rectified_img_r, key_pts_r, None, color=(0,255,0))

    cv2.imshow('Izquierda - FAST keypoints', img_key_pts_l)
    cv2.imshow('Derecha - FAST keypoints', img_key_pts_r)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(f'{output_dir}/keypoints_left.png', img_key_pts_l)
    cv2.imwrite(f'{output_dir}/keypoints_right.png', img_key_pts_r)

    # Ejercicio C

    # Obtenemos todos los matches
    all_matches = script.getMatches()

    # Filtramos los matches buenos con distancia mayor a 30
    good_matches = [m for m in all_matches if m.distance < 30]

    # Imprimimos los resultados
    img_all_matches = cv2.drawMatches(rectified_img_l, key_pts_l, img_r, key_pts_r, all_matches, None)
    img_good_matches = cv2.drawMatches(rectified_img_r, key_pts_l, img_r, key_pts_r, good_matches, None)

    cv2.imshow("Todos los matches", img_all_matches)
    cv2.imshow("Matches buenos con distancia < 30", img_good_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(f'{output_dir}/all_matches.png', img_all_matches)
    cv2.imwrite(f'{output_dir}/good_matches.png', img_good_matches)

    # Ejercicio D

    good_pts_l, good_pts_r = script.getMatchesPoints(good_matches)
    good_pts_3D = script.triangulate(good_pts_l, good_pts_r)

    # Escribimos el array de puntos 3D en un archivo .npy para luego visualizarlo en rviz
    np.save(f'{output_dir}/points3D.npy', good_pts_3D)

    # Ejercicio E

    filtered_matches = script.filterMatches(good_matches, good_pts_l, good_pts_r)

    # Genermoas la imagen con los matches filtrados
    img_filtered_matches = cv2.drawMatches(rectified_img_l, key_pts_l, rectified_img_r, key_pts_r, filtered_matches, None)

    cv2.imshow("Matches espureos filtrados", img_filtered_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(f'{output_dir}/filtered_matches.png', img_filtered_matches)

    # Transformamos los puntos a la imagen derecha utilizando la matriz M
    filtered_pts_l, filtered_pts_r = script.getMatchesPoints(filtered_matches)
    filtered_pts_transformed_r = script.transformPoints(filtered_pts_l)

    # Dibujamos círculos verdes para los puntos transformados
    img_transformed = img_r.copy()
    for pt in filtered_pts_transformed_r:
        x, y = pt[0]
        cv2.circle(img_transformed, (int(x), int(y)), 4, (0, 255, 0), -1)

    cv2.imshow("Puntos transformados en la imagen derecha", img_transformed)
    cv2.imwrite(f'{output_dir}/transformed_points_r.png', img_transformed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Ejercicio F

    world_good_pts_3d = script.to_world(good_pts_3D)
    np.save(f'{output_dir}/world_good_points_3d.npy', world_good_pts_3d)

    # Ejercicio G
    
    disparity_map = script.computeDisparityMap(rectified_img_l, rectified_img_r)

    cv2.imshow('Mapa de Disparidad', disparity_map)
    cv2.imwrite(f'{output_dir}/disparity_map.png', disparity_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Ejercicio H
    rebuilded_pts3D = script.rebuildDense3DScene()
    np.save(f'{output_dir}/rebuilded_points3D.npy', rebuilded_pts3D)

    # Ejercicio I

    world_rebuilded_pts_3d = script.to_world(rebuilded_pts3D)
    np.save(f'{output_dir}/world_rebuilded_points_3d.npy', world_rebuilded_pts_3d)

    # Ejercicio J

    # Agregamos a la imagen izquierda los poses de las camaras
    img_poses = img_r.copy()

    camera_pose_l, camera_pose_r = script.stimatePose(filtered_pts_l, filtered_pts_r)

    # Dibujamos los centros de las camaras
    cv2.circle(img_poses, camera_pose_l, 8, (0,0,255), -1)
    cv2.circle(img_poses, camera_pose_r, 8, (255,0,0), -1)

    # Dibujamos una baseline
    cv2.line(img_poses, camera_pose_l, camera_pose_r, (0,0,0), 2, cv2.LINE_AA)

    # Mostrar y guardar
    cv2.imshow("Stereo Poses", img_poses)
    cv2.imwrite(f'{output_dir}/cameras_pose.png', img_poses)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()