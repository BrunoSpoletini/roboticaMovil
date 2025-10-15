import cv2
import paths

# --- Cargar imágenes izquierda y derecha ---
left_img = cv2.imread(f'{paths.folder_path}/TP3/images/left.png', cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread(f'{paths.folder_path}/TP3/images/right.png', cv2.IMREAD_GRAYSCALE)

# --- Detector FAST ---
fast = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)

# Detectar keypoints
kp_left = fast.detect(left_img, None)
kp_right = fast.detect(right_img, None)

# --- Descriptor BRIEF ---
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

# Calcular descriptores para los keypoints
kp_left, des_left = brief.compute(left_img, kp_left)
kp_right, des_right = brief.compute(right_img, kp_right)

print(f"FAST detectó {len(kp_left)} puntos en la izquierda y {len(kp_right)} en la derecha.")
print(f"Descriptores: {des_left.shape} (izquierda), {des_right.shape} (derecha)")

# --- Dibujar los keypoints ---
img_left_kp = cv2.drawKeypoints(left_img, kp_left, None, color=(0,255,0))
img_right_kp = cv2.drawKeypoints(right_img, kp_right, None, color=(0,255,0))

# --- Mostrar resultados ---
cv2.imshow('Izquierda - FAST keypoints', img_left_kp)
cv2.imshow('Derecha - FAST keypoints', img_right_kp)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --- Guardar imágenes para el informe ---
cv2.imwrite(f'{paths.folder_path}/TP3/left_FAST_features.png', img_left_kp)
cv2.imwrite(f'{paths.folder_path}/TP3/right_FAST_features.png', img_right_kp)
