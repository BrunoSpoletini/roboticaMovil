import cv2
import numpy as np
import yaml

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

    cv2.imwrite('/{dir_out}/left_keypoints.png', img_left_kp)
    cv2.imwrite('/{dir_out}/right_keypoints.png', img_right_kp)

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

def triangulate(kp_left, kp_right, good_matches):

    P1 = load_calib('/home/sco/roboticaMovil/TP3/calibrationData/left.yaml')
    P2 = load_calib('/home/sco/roboticaMovil/TP3/calibrationData/right.yaml')

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
    P = np.array(data['projection_matrix']['data']).reshape(3,4)
    return P

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
    left_img = cv2.imread('/{dir_in}/left_keypoints.png', cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread('/{dir_in}/right_keypoints.png', cv2.IMREAD_GRAYSCALE)

    kp_left, kp_right, des_left, des_right = getFeatures(dir_out, left_img, right_img)




if __name__ == '__main__':
    main()
