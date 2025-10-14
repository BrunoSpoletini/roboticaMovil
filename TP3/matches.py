import cv2

# --- Cargar imágenes en escala de grises ---
img1 = cv2.imread("left_FAST_features.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("right_FAST_features.png", cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("Error: no se pudieron cargar las imágenes.")
    exit()

# --- Detectar y describir features con ORB ---
orb = cv2.ORB_create(1000)
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

# --- Crear el matcher (Brute Force con Hamming) ---
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# --- Encontrar los matches ---
matches = bf.match(descriptors1, descriptors2)

# --- Ordenar matches por distancia (opcional, solo para visualización más limpia) ---
matches = sorted(matches, key=lambda x: x.distance)

# --- Visualizar TODOS los matches ---
img_matches_all = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None)
cv2.imshow("Todos los matches", img_matches_all)
cv2.imwrite("matches_todos.png", img_matches_all)

# --- Filtrar matches con distancia < 30 ---
good_matches = [m for m in matches if m.distance < 30]

# --- Visualizar solo los buenos matches ---
img_matches_good = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None)
cv2.imshow("Matches con distancia < 30", img_matches_good)
cv2.imwrite("matches_buenos.png", img_matches_good)

print(f"Total de matches: {len(matches)}")
print(f"Matches con distancia < 30: {len(good_matches)}")

cv2.waitKey(0)
cv2.destroyAllWindows()
