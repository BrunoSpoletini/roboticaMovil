import cv2
import numpy as np
import yaml

output_dir = './resultados'

path_camara_left = './images/left.png'
path_camara_right = './images/right.png'

path_calibracion_left = './calibrationData/left.yaml'
path_calibracion_right = './calibrationData/right.yaml'


def rectify(left_image, right_image, leftCameraInfo, rightCameraInfo):
    # Obtener las dimensiones de las imágenes
    image_size = (leftCameraInfo['image_width'], leftCameraInfo['image_height'])

    # Extraer matrices de rotacion y traslacion entre camaras
    K = np.array(rightCameraInfo['camera_matrix']['data']).reshape(3, 3)
    P = np.array(rightCameraInfo['projection_matrix']['data']).reshape(3, 4)

    # Compute [R | T] = K^-1 * P
    K_inv = np.linalg.inv(K)
    RT = np.dot(K_inv, P)

    # Extract R (3x3) and T (3x1)
    R = RT[:, :3]
    T = RT[:, 3]

    # Calcular las matrices de rectificación
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        cameraMatrix1=np.array(leftCameraInfo['camera_matrix']['data']).reshape(3, 3),
        distCoeffs1=np.array(leftCameraInfo['distortion_coefficients']['data']),
        cameraMatrix2=np.array(rightCameraInfo['camera_matrix']['data']).reshape(3, 3),
        distCoeffs2=np.array(rightCameraInfo['distortion_coefficients']['data']),
        imageSize=(leftCameraInfo['image_width'], leftCameraInfo['image_height']),
        R=np.array(R),
        T=np.array(T)
    )

    # Calcular los mapas de remapeo para las cámaras izquierda y derecha
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        cameraMatrix=np.array(leftCameraInfo['camera_matrix']['data']).reshape(3, 3),
        distCoeffs=np.array(leftCameraInfo['distortion_coefficients']['data']),
        R=R1,
        newCameraMatrix=P1,
        size=image_size,
        m1type=cv2.CV_32FC1
    )

    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        cameraMatrix=np.array(rightCameraInfo['camera_matrix']['data']).reshape(3, 3),
        distCoeffs=np.array(rightCameraInfo['distortion_coefficients']['data']),
        R=R2,
        newCameraMatrix=P2,
        size=image_size,
        m1type=cv2.CV_32FC1
    )

    # Aplicar remapeo a las imágenes
    rectified_left = cv2.remap(left_image, left_map1, left_map2, interpolation=cv2.INTER_LINEAR)
    rectified_right = cv2.remap(right_image, right_map1, right_map2, interpolation=cv2.INTER_LINEAR)

    return rectified_left, rectified_right

def getFeatures(dir_out, left_img, right_img):

    # Obtenemos los keypoints
    fast = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
    kp_left = fast.detect(left_img, None)
    kp_right = fast.detect(right_img, None)

    # Obtenemos los descriptores
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    kp_left, des_left = brief.compute(left_img, kp_left)
    kp_right, des_right = brief.compute(right_img, kp_right)

    # Imprimimos los resultados
    img_left_kp = cv2.drawKeypoints(left_img, kp_left, None, color=(0,255,0))
    img_right_kp = cv2.drawKeypoints(right_img, kp_right, None, color=(0,255,0))

    cv2.imshow('Izquierda - FAST keypoints', img_left_kp)
    cv2.imshow('Derecha - FAST keypoints', img_right_kp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(f'{output_dir}/left_keypoints.png', img_left_kp)
    cv2.imwrite(f'{output_dir}/right_keypoints.png', img_right_kp)

    return kp_left, kp_right, des_left, des_right

def getMatches(left_img, right_img, kp_left, kp_right, des_left, des_right):

    # Generamos el Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Encontramos los matches
    all_matches = bf.match(des_left, des_right)

    # Filtramos los matches posibles con distancia mayor a 30
    le_matches = [m for m in all_matches if m.distance < 30]

    # Imprimimos los resultados
    img_all_matches = cv2.drawMatches(left_img, kp_left, right_img, kp_right, all_matches, None)
    img_le_matches = cv2.drawMatches(left_img, kp_left, right_img, kp_right, le_matches, None)

    cv2.imshow("Todos los matches", img_all_matches)
    cv2.imshow("Matches con distancia < 30", img_le_matches)

    cv2.imwrite("matches_todos.png", img_all_matches)
    cv2.imwrite("matches_dist_menor_30.png", img_le_matches)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return all_matches, le_matches

def triangulate(kp_left, kp_right, good_matches, leftCameraInfo, rightCameraInfo):

    P1 = np.array(leftCameraInfo['projection_matrix']['data']).reshape(3,4)
    P2 = np.array(rightCameraInfo['projection_matrix']['data']).reshape(3,4)

    # Ordenamos los puntos ordenamos por match
    src_pts = np.float32([kp_left[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([kp_right[m.trainIdx].pt for m in good_matches])

    # Realizamos la triangulacion
    points4D = cv2.sfm.triangulatePoints(P1, P2, src_pts.T, dst_pts.T)
    points3D = (points4D[:3] / points4D[3]).T

    return points3D

def load_calib(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return data

def filterMatches(good_matches, left_img, right_img, src_pts, dst_pts):

    if len(good_matches) > 10:
    
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
    
        h,w = left_img.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
    
        right_img = cv2.polylines(right_img,[np.int32(dst)],True,255,3, cv2.LINE_AA)
 
    else:
        print( "Not enough matches are found - {}/{}".format(len(good_matches), 10) )
        matchesMask = None    

def computeDisparityMap(left_image, right_image):
    disparity = cv2.stereo_matcher_object.compute(left_image, right_image)
    return disparity

def rebuild3DScene(disparity):
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(M1, d1, M2, d2, image_size, R, T, cv2.CALIB_ZERO_DISPARITY, alpha=0)
    points3D = cv2.reprojectImageTo3D(disparity, Q)
    return points3D


def stimatePose():
    return

def main(args=None):

    # Cargamos las imágenes izquierda y derecha
    left_img = cv2.imread(path_camara_left, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(path_camara_right, cv2.IMREAD_GRAYSCALE)

    # Cargamos las calibraciones
    leftCameraInfo = load_calib(path_calibracion_left)
    rightCameraInfo = load_calib(path_calibracion_right)

    # Obtener las imagenes rectificadas
    rectified_left, rectified_right = rectify(left_img, right_img, leftCameraInfo, rightCameraInfo)




    # Obtenemos los features y descriptores
    kp_left, kp_right, des_left, des_right = getFeatures(output_dir, rectified_left, rectified_right)

#     # Obtenemos los matches
#     all_matches, le_matches = getMatches(rectified_left, rectified_right, kp_left, kp_right, des_left, des_right)

#     # Realizamos la triangulación
#     points3D = triangulate(kp_left, kp_right, le_matches, leftCameraInfo, rightCameraInfo)

if __name__ == '__main__':
    main()


    # # Mostrar las imágenes rectificadas
    # cv2.imshow('Rectified Left', rectified_left)
    # cv2.imshow('Rectified Right', rectified_right)
    # cv2.waitKey(10000)
    # cv2.destroyAllWindows()