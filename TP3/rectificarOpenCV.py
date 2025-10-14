# import csv
import yaml
import numpy as np
import cv2 as cv

def leer_camera_info():
    # Leer datos de calibracion de camaras estereo
    with open("./calibrationData/left.yaml", "r") as f:
        leftCameraInfo = yaml.safe_load(f)

    with open("./calibrationData/right.yaml", "r") as f:
        rightCameraInfo = yaml.safe_load(f)

    print("Left Camera Calibration Data:")
    for key, value in leftCameraInfo.items():
        print(f"  {key}: {value}")

    print("Right Camera Calibration Data:")
    for key, value in rightCameraInfo.items():
        print(f"  {key}: {value}")
    return leftCameraInfo, rightCameraInfo


def extraer_rt(K, P):
    K = np.array(K).reshape(3, 3)
    P = np.array(P).reshape(3, 4)

    # Compute [R | T] = K^-1 * P
    K_inv = np.linalg.inv(K)
    RT = np.dot(K_inv, P)

    # Extract R (3x3) and T (3x1)
    R = RT[:, :3]
    T = RT[:, 3]

    return R, T



def leer_imagenes():
    left_image_path = './images/cam0.png'
    right_image_path = './images/cam1.png'

    left_image = cv.imread(left_image_path, cv.IMREAD_COLOR)
    right_image = cv.imread(right_image_path, cv.IMREAD_COLOR)

    return left_image, right_image


def rectificar_imagenes(left_image, right_image, leftCameraInfo, rightCameraInfo, R1, R2, P1, P2):
    # Obtener las dimensiones de las imágenes
    image_size = (leftCameraInfo['image_width'], leftCameraInfo['image_height'])

    # Calcular los mapas de remapeo para las cámaras izquierda y derecha
    left_map1, left_map2 = cv.initUndistortRectifyMap(
        cameraMatrix=np.array(leftCameraInfo['camera_matrix']['data']).reshape(3, 3),
        distCoeffs=np.array(leftCameraInfo['distortion_coefficients']['data']),
        R=R1,
        newCameraMatrix=P1,
        size=image_size,
        m1type=cv.CV_32FC1
    )

    right_map1, right_map2 = cv.initUndistortRectifyMap(
        cameraMatrix=np.array(rightCameraInfo['camera_matrix']['data']).reshape(3, 3),
        distCoeffs=np.array(rightCameraInfo['distortion_coefficients']['data']),
        R=R2,
        newCameraMatrix=P2,
        size=image_size,
        m1type=cv.CV_32FC1
    )

    # Aplicar remapeo a las imágenes
    rectified_left = cv.remap(left_image, left_map1, left_map2, interpolation=cv.INTER_LINEAR)
    rectified_right = cv.remap(right_image, right_map1, right_map2, interpolation=cv.INTER_LINEAR)

    return rectified_left, rectified_right


def main():
    leftCameraInfo, rightCameraInfo = leer_camera_info()
    left_image, right_image = leer_imagenes()

    # Extraer matrices de rotacion y traslacion entre camaras
    K = rightCameraInfo['camera_matrix']['data']
    P = rightCameraInfo['projection_matrix']['data']
    R, T = extraer_rt(K, P)

    # Calcular las matrices de rectificación
    R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(
        cameraMatrix1=np.array(leftCameraInfo['camera_matrix']['data']).reshape(3, 3),
        distCoeffs1=np.array(leftCameraInfo['distortion_coefficients']['data']),
        cameraMatrix2=np.array(rightCameraInfo['camera_matrix']['data']).reshape(3, 3),
        distCoeffs2=np.array(rightCameraInfo['distortion_coefficients']['data']),
        imageSize=(leftCameraInfo['image_width'], leftCameraInfo['image_height']),
        R=np.array(R),
        T=np.array(T)
    )

    # Rectificar las imágenes
    rectified_left, rectified_right = rectificar_imagenes(
        left_image, right_image, leftCameraInfo, rightCameraInfo, R1, R2, P1, P2
    )

    # Mostrar las imágenes rectificadas
    cv.imshow('Rectified Left', rectified_left)
    cv.imshow('Rectified Right', rectified_right)
    cv.waitKey(10000)
    cv.destroyAllWindows()

    # Guardar las imágenes rectificadas
    cv.imwrite('./images/rectified_left.png', rectified_left)
    cv.imwrite('./images/rectified_right.png', rectified_right)
    
main()