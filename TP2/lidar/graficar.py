#!/usr/bin/env python3
"""
Script para graficar datos de LIDAR
Lee datos de un archivo de texto y genera gráficos polares y cartesianos
"""

import numpy as np
import matplotlib.pyplot as plt
import re
import sys
import os


def parse_lidar_data(file_path):
    """
    Parsea los datos del archivo de lidar y extrae los parámetros necesarios
    """
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


def plot_lidar_data(data, show_plot=True, save_plot=True):
    """
    Grafica los datos del lidar en coordenadas cartesianas
    """
    ranges = np.array(data["ranges"])

    # Generar ángulos
    num_points = len(ranges)
    angles = np.linspace(
        data["angle_min"],
        data["angle_min"] + num_points * data["angle_increment"],
        num_points,
    )

    # Filtrar valores infinitos y fuera de rango
    valid_mask = (
        (ranges < data["range_max"])
        & (ranges > data["range_min"])
        & np.isfinite(ranges)
    )
    valid_ranges = ranges[valid_mask]
    valid_angles = angles[valid_mask]

    # Convertir a coordenadas cartesianas
    x = valid_ranges * np.cos(valid_angles)
    y = valid_ranges * np.sin(valid_angles)

    # Crear figura
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Gráfico cartesiano
    ax.scatter(x, y, c="blue", s=2, alpha=0.7)
    ax.set_title("Datos LIDAR - Coordenadas Cartesianas")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    # Marcar la posición del robot
    ax.plot(0, 0, "ro", markersize=10, label="Robot")
    ax.legend()

    plt.tight_layout()

    # Mostrar estadísticas
    print("\n=== Estadísticas de los datos LIDAR ===")
    print(f"Número total de mediciones: {len(ranges)}")
    print(f"Mediciones válidas: {len(valid_ranges)}")
    print(f"Rango de ángulos: {data['angle_min']:.3f} - {data['angle_max']:.3f} rad")
    print(
        f"Incremento angular: {data['angle_increment']:.6f} rad ({np.degrees(data['angle_increment']):.3f}°)"
    )
    print(f"Distancia mínima detectada: {np.min(valid_ranges):.3f} m")
    print(f"Distancia máxima detectada: {np.max(valid_ranges):.3f} m")
    print(f"Distancia promedio: {np.mean(valid_ranges):.3f} m")

    if save_plot:
        plt.savefig("lidar_plot.png", dpi=300, bbox_inches="tight")
        print("\nGráfico guardado como 'lidar_plot.png'")

    if show_plot:
        plt.show()

    return fig, ax


def main():
    """
    Función principal
    """
    # Buscar archivo de datos
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Buscar en el directorio actual
        possible_files = ["header.txt", "../header.txt", "../../header.txt"]
        file_path = None
        for f in possible_files:
            if os.path.exists(f):
                file_path = f
                break

        if file_path is None:
            print("No se encontró archivo de datos. Uso:")
            print("python graficar.py [archivo_datos.txt]")
            return

    if not os.path.exists(file_path):
        print(f"Error: No se encontró el archivo {file_path}")
        return

    print(f"Procesando archivo: {file_path}")

    try:
        # Parsear datos
        data = parse_lidar_data(file_path)

        # Crear gráfico
        print("Generando gráfico...")
        plot_lidar_data(data)

        print("\n¡Gráfico generado exitosamente!")

    except Exception as e:
        print(f"Error al procesar los datos: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
