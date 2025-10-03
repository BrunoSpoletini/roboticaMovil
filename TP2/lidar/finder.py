#!/usr/bin/env python3
"""
Script para detectar cilindros en datos de LIDAR
Divide los datos en conjuntos de puntos que representen cilindros
y calcula el centro de cada cilindro detectado
"""

import numpy as np
import matplotlib.pyplot as plt
import re
import sys
import os
from sklearn.cluster import DBSCAN


def parse_lidar_data(file_path):
    with open(file_path, "r") as f:
        content = f.read()

    # Extraer parámetros del header
    angle_min = float(re.search(r"angle_min:\s*([\d.-]+)", content).group(1))
    angle_max = float(re.search(r"angle_max:\s*([\d.-]+)", content).group(1))
    angle_increment = float(
        re.search(r"angle_increment:\s*([\d.-]+)", content).group(1)
    )
    range_min = float(re.search(r"range_min:\s*([\d.e-]+)", content).group(1))
    range_max = float(re.search(r"range_max:\s*([\d.-]+)", content).group(1))

    # Extraer rangos
    ranges_section = re.search(
        r"ranges:\s*\n(.*?)(?:\nintensities|\n\w+:|$)", content, re.DOTALL
    )
    if ranges_section:
        ranges_text = ranges_section.group(1)
        # Extraer números y manejar .inf
        ranges = []
        for line in ranges_text.split("\n"):
            line = line.strip()
            if line.startswith("-"):
                value = line[1:].strip()
                if value == ".inf":
                    ranges.append(np.inf)
                elif value == "...":
                    break
                else:
                    try:
                        ranges.append(float(value))
                    except ValueError:
                        continue
    return {
        "angle_min": angle_min,
        "angle_max": angle_max,
        "angle_increment": angle_increment,
        "range_min": range_min,
        "range_max": range_max,
        "ranges": ranges,
    }


def convert_to_cartesian(data):
    ranges = np.array(data["ranges"])

    # Generar ángulos
    num_points = len(ranges)
    angles = np.linspace(
        data["angle_min"],
        data["angle_min"] + num_points * data["angle_increment"],
        num_points,
    )

    # Filtrar valores infinitos y fuera de rango (evitamos escanear las "paredes")
    valid_mask = (ranges < 10) & (ranges > data["range_min"]) & np.isfinite(ranges)
    valid_ranges = ranges[valid_mask]
    valid_angles = angles[valid_mask]

    # Convertir a coordenadas cartesianas
    x = valid_ranges * np.cos(valid_angles)
    y = valid_ranges * np.sin(valid_angles)

    return np.column_stack((x, y)), valid_ranges, valid_angles


# def detect_cylinders(points, tolerance=0.5, min_points=5):
#     """
#     Detecta cilindros en los puntos usando clustering DBSCAN

#     Args:
#         points: Array de puntos (x, y) en coordenadas cartesianas
#         tolerance: Tolerancia en metros para considerar puntos del mismo cilindro
#         min_points: Número mínimo de puntos para considerar un cluster válido

#     Returns:
#         labels: Etiquetas de cluster para cada punto (-1 = ruido)
#         n_clusters: Número de clusters detectados
#     """
#     # Usar DBSCAN para clustering
#     clustering = DBSCAN(eps=tolerance, min_samples=min_points)
#     labels = clustering.fit_predict(points)

#     n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

#     return labels, n_clusters


# def analyze_cylinder_cluster(points, ranges, angles):
#     """
#     Analiza un cluster de puntos de un cilindro y calcula estadísticas

#     Args:
#         points: Puntos del cluster en coordenadas cartesianas
#         ranges: Distancias correspondientes
#         angles: Ángulos correspondientes

#     Returns:
#         dict con estadísticas del cilindro
#     """
#     if len(points) == 0:
#         return None

#     # Encontrar índices de min, media y max en distancia
#     min_idx = np.argmin(ranges)
#     max_idx = np.argmax(ranges)
#     median_idx = np.argsort(ranges)[len(ranges) // 2]

#     stats = {
#         "n_points": len(points),
#         "min_point": {
#             "cartesian": points[min_idx],
#             "range": ranges[min_idx],
#             "angle": angles[min_idx],
#         },
#         "median_point": {
#             "cartesian": points[median_idx],
#             "range": ranges[median_idx],
#             "angle": angles[median_idx],
#         },
#         "max_point": {
#             "cartesian": points[max_idx],
#             "range": ranges[max_idx],
#             "angle": angles[max_idx],
#         },
#         "centroid": np.mean(points, axis=0),
#         "std_dev": np.std(points, axis=0),
#     }

#     return stats


# def calculate_cylinder_center(
#     min_point, median_point, max_point, method="circumcenter"
# ):
#     """
#     Calcula el centro del cilindro usando diferentes métodos

#     Args:
#         min_point, median_point, max_point: Coordenadas cartesianas de los tres puntos
#         method: 'circumcenter', 'centroid', 'least_squares'

#     Returns:
#         Coordenadas del centro estimado del cilindro
#     """
#     points = np.array([min_point, median_point, max_point])

#     if method == "centroid":
#         # Simple centroide de los tres puntos
#         return np.mean(points, axis=0)

#     elif method == "circumcenter":
#         # Centro del círculo que pasa por los tres puntos
#         return calculate_circumcenter(min_point, median_point, max_point)

#     elif method == "least_squares":
#         # Ajuste por mínimos cuadrados (requiere más puntos)
#         return calculate_least_squares_center(points)

#     else:
#         return np.mean(points, axis=0)


# def calculate_circumcenter(p1, p2, p3):
#     """
#     Calcula el circumcentro de tres puntos (centro del círculo que pasa por los tres puntos)
#     """
#     ax, ay = p1
#     bx, by = p2
#     cx, cy = p3

#     # Fórmula del circumcentro
#     d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))

#     if abs(d) < 1e-10:  # Puntos colineales
#         return np.mean([p1, p2, p3], axis=0)

#     ux = (
#         (ax**2 + ay**2) * (by - cy)
#         + (bx**2 + by**2) * (cy - ay)
#         + (cx**2 + cy**2) * (ay - by)
#     ) / d
#     uy = (
#         (ax**2 + ay**2) * (cx - bx)
#         + (bx**2 + by**2) * (ax - cx)
#         + (cx**2 + cy**2) * (bx - ax)
#     ) / d

#     return np.array([ux, uy])


# def calculate_least_squares_center(points):
#     """
#     Calcula el centro del círculo usando mínimos cuadrados
#     """
#     if len(points) < 3:
#         return np.mean(points, axis=0)

#     # Centroide inicial
#     x_mean = np.mean(points[:, 0])
#     y_mean = np.mean(points[:, 1])

#     # Resolver sistema por mínimos cuadrados
#     # (x - cx)² + (y - cy)² = r²
#     # Expandido: x² + y² - 2*cx*x - 2*cy*y + cx² + cy² - r² = 0

#     A = np.column_stack([2 * points[:, 0], 2 * points[:, 1], np.ones(len(points))])
#     b = points[:, 0] ** 2 + points[:, 1] ** 2

#     try:
#         result = np.linalg.lstsq(A, b, rcond=None)[0]
#         center = result[:2]
#         return center
#     except np.linalg.LinAlgError:
#         return np.array([x_mean, y_mean])


# def plot_cylinder_detection(points, labels, cylinder_stats, cylinder_centers):
#     """
#     Visualiza la detección de cilindros
#     """
#     fig, ax = plt.subplots(1, 1, figsize=(12, 10))

#     # Colores para diferentes clusters
#     colors = plt.cm.Set1(np.linspace(0, 1, max(len(cylinder_stats), 1)))

#     # Plotear puntos de ruido
#     noise_mask = labels == -1
#     if np.any(noise_mask):
#         ax.scatter(
#             points[noise_mask, 0],
#             points[noise_mask, 1],
#             c="gray",
#             s=10,
#             alpha=0.5,
#             label="Ruido",
#         )

#     # Plotear cada cluster de cilindro
#     for i, (cluster_id, stats) in enumerate(cylinder_stats.items()):
#         cluster_mask = labels == cluster_id
#         cluster_points = points[cluster_mask]

#         # Puntos del cluster
#         ax.scatter(
#             cluster_points[:, 0],
#             cluster_points[:, 1],
#             c=[colors[i]],
#             s=20,
#             alpha=0.7,
#             label=f"Cilindro {cluster_id + 1}",
#         )

#         # Puntos especiales (min, median, max)
#         min_pt = stats["min_point"]["cartesian"]
#         median_pt = stats["median_point"]["cartesian"]
#         max_pt = stats["max_point"]["cartesian"]

#         ax.plot(
#             min_pt[0],
#             min_pt[1],
#             "g^",
#             markersize=10,
#             label=f"Min {cluster_id + 1}" if i == 0 else "",
#         )
#         ax.plot(
#             median_pt[0],
#             median_pt[1],
#             "y*",
#             markersize=12,
#             label=f"Median {cluster_id + 1}" if i == 0 else "",
#         )
#         ax.plot(
#             max_pt[0],
#             max_pt[1],
#             "rv",
#             markersize=10,
#             label=f"Max {cluster_id + 1}" if i == 0 else "",
#         )

#         # Centro del cilindro
#         center = cylinder_centers[cluster_id]
#         ax.plot(
#             center[0],
#             center[1],
#             "ko",
#             markersize=12,
#             markerfacecolor=colors[i],
#             markeredgecolor="black",
#             linewidth=2,
#             label=f"Centro {cluster_id + 1}",
#         )

#         # Círculo estimado del cilindro
#         radius = np.mean(np.linalg.norm(cluster_points - center, axis=1))
#         circle = plt.Circle(
#             center, radius, fill=False, color=colors[i], linestyle="--", alpha=0.7
#         )
#         ax.add_patch(circle)

#     # Posición del robot
#     ax.plot(0, 0, "ro", markersize=15, label="Robot")

#     ax.set_xlabel("X (m)")
#     ax.set_ylabel("Y (m)")
#     ax.set_title("Detección de Cilindros en datos LIDAR")
#     ax.grid(True, alpha=0.3)
#     ax.axis("equal")
#     ax.legend()

#     plt.tight_layout()
#     plt.savefig("cylinder_detection.png", dpi=300, bbox_inches="tight")
#     print("Gráfico de detección guardado como 'cylinder_detection.png'")
#     plt.show()


# def main():
#     """
#     Función principal
#     """
#     # Buscar archivo de datos
#     if len(sys.argv) > 1:
#         file_path = sys.argv[1]
#     else:
#         # Buscar en el directorio actual
#         possible_files = ["header.txt", "../header.txt", "../../header.txt"]
#         file_path = None
#         for f in possible_files:
#             if os.path.exists(f):
#                 file_path = f
#                 break

#         if file_path is None:
#             print("No se encontró archivo de datos. Uso:")
#             print("python finder.py [archivo_datos.txt]")
#             return

#     if not os.path.exists(file_path):
#         print(f"Error: No se encontró el archivo {file_path}")
#         return

#     print(f"Procesando archivo: {file_path}")

#     try:
#         # Parsear datos
#         data = parse_lidar_data(file_path)

#         # Convertir a coordenadas cartesianas
#         points, ranges, angles = convert_to_cartesian(data)
#         print(f"Puntos válidos encontrados: {len(points)}")

#         # Detectar cilindros
#         tolerance = 0.5  # 0.5 metros de tolerancia
#         labels, n_clusters = detect_cylinders(points, tolerance=tolerance)
#         print(f"Cilindros detectados: {n_clusters}")

#         if n_clusters == 0:
#             print("No se detectaron cilindros con la tolerancia especificada.")
#             return

#         # Analizar cada cilindro
#         cylinder_stats = {}
#         cylinder_centers = {}

#         for cluster_id in range(n_clusters):
#             # Obtener puntos del cluster
#             cluster_mask = labels == cluster_id
#             cluster_points = points[cluster_mask]
#             cluster_ranges = ranges[cluster_mask]
#             cluster_angles = angles[cluster_mask]

#             # Analizar cluster
#             stats = analyze_cylinder_cluster(
#                 cluster_points, cluster_ranges, cluster_angles
#             )
#             if stats is not None:
#                 cylinder_stats[cluster_id] = stats

#                 # Calcular centro del cilindro
#                 min_pt = stats["min_point"]["cartesian"]
#                 median_pt = stats["median_point"]["cartesian"]
#                 max_pt = stats["max_point"]["cartesian"]

#                 center = calculate_cylinder_center(
#                     min_pt, median_pt, max_pt, method="circumcenter"
#                 )
#                 cylinder_centers[cluster_id] = center

#                 # Mostrar resultados
#                 print(f"\n=== Cilindro {cluster_id + 1} ===")
#                 print(f"Número de puntos: {stats['n_points']}")
#                 print(
#                     f"Punto mínimo (dist): ({min_pt[0]:.3f}, {min_pt[1]:.3f}) - {stats['min_point']['range']:.3f}m"
#                 )
#                 print(
#                     f"Punto mediano (dist): ({median_pt[0]:.3f}, {median_pt[1]:.3f}) - {stats['median_point']['range']:.3f}m"
#                 )
#                 print(
#                     f"Punto máximo (dist): ({max_pt[0]:.3f}, {max_pt[1]:.3f}) - {stats['max_point']['range']:.3f}m"
#                 )
#                 print(f"Centro estimado: ({center[0]:.3f}, {center[1]:.3f})")
#                 print(
#                     f"Centroide: ({stats['centroid'][0]:.3f}, {stats['centroid'][1]:.3f})"
#                 )

#         # Visualizar resultados
#         if cylinder_stats:
#             plot_cylinder_detection(points, labels, cylinder_stats, cylinder_centers)

#         print(f"\n¡Detección completada! {len(cylinder_stats)} cilindros analizados.")

#     except Exception as e:
#         print(f"Error al procesar los datos: {e}")
#         import traceback

#         traceback.print_exc()


# if __name__ == "__main__":
#     main()
