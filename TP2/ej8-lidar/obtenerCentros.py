#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import re


def parse_data(file):
    with open(file, "r") as f:
        contenido = f.read()

    # Extraer parámetros del header
    angulo_min = float(re.search(r"angle_min:\s*([\d.-]+)", contenido).group(1))
    angulo_max = float(re.search(r"angle_max:\s*([\d.-]+)", contenido).group(1))
    incremento_angulo = float(
        re.search(r"angle_increment:\s*([\d.-]+)", contenido).group(1)
    )
    rango_min = float(re.search(r"range_min:\s*([\d.e-]+)", contenido).group(1))
    rango_max = float(re.search(r"range_max:\s*([\d.-]+)", contenido).group(1))

    # Extraer rangos
    seccion_rangos = re.search(
        r"ranges:\s*\n(.*?)(?:\nintensities|\n\w+:|$)", contenido, re.DOTALL
    )
    if seccion_rangos:
        texto_rangos = seccion_rangos.group(1)
        rangos = []
        for linea in texto_rangos.split("\n"):
            linea = linea.strip()
            if linea.startswith("-"):
                valor = linea[1:].strip()
                if valor == ".inf":
                    rangos.append(np.inf)
                elif valor == "...":
                    break
                else:
                    try:
                        rangos.append(float(valor))
                    except ValueError:
                        continue
    return {
        "angulo_min": angulo_min,
        "angulo_max": angulo_max,
        "incremento_angulo": incremento_angulo,
        "rango_min": rango_min,
        "rango_max": rango_max,
        "rangos": rangos,
    }


# Convertimos a coordenadas cartesianas y filtramos las paredes invisibles
def convertir_a_cartesiano(datos):
    rangos = np.array(datos["rangos"])

    # Generar ángulos
    num_puntos = len(rangos)
    angulos = np.linspace(
        datos["angulo_min"],
        datos["angulo_min"] + num_puntos * datos["incremento_angulo"],
        num_puntos,
    )

    # Filtrar valores infinitos y fuera de rango (evitamos escanear las "paredes")
    mascara_valida = (rangos < 10) & (rangos > datos["rango_min"]) & np.isfinite(rangos)
    rangos_validos = rangos[mascara_valida]
    angulos_validos = angulos[mascara_valida]

    # Convertir a coordenadas cartesianas
    x = rangos_validos * np.cos(angulos_validos)
    y = rangos_validos * np.sin(angulos_validos)

    return np.column_stack((x, y)), rangos_validos, angulos_validos


def graficar_puntos_cartesianos(puntos):
    plt.figure(figsize=(10, 8))
    plt.scatter(puntos[:, 0], puntos[:, 1], c="blue", s=5, alpha=0.7, label="Puntos")
    plt.title("Puntos en Coordenadas Cartesianas")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.show()


def agrupar_puntos_por_distancia(puntos, umbral=0.5):
    if len(puntos) == 0:
        return []

    grupos = []
    grupo_actual = [puntos[0]]

    for i in range(1, len(puntos)):
        distancia = np.linalg.norm(puntos[i] - puntos[i - 1])
        if distancia <= umbral:
            grupo_actual.append(puntos[i])
        else:
            grupos.append(np.array(grupo_actual))
            grupo_actual = [puntos[i]]

    # Agregar el último grupo
    if grupo_actual:
        grupos.append(np.array(grupo_actual))

    return grupos


# Cálculo del centro del círculo que pasa por tres puntos
def centro_circulo(p1, p2, p3):
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3

    D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if D == 0:
        raise ValueError("Los tres puntos son colineales, no definen un círculo.")

    ux = (
        (x1**2 + y1**2) * (y2 - y3)
        + (x2**2 + y2**2) * (y3 - y1)
        + (x3**2 + y3**2) * (y1 - y2)
    ) / D

    uy = (
        (x1**2 + y1**2) * (x3 - x2)
        + (x2**2 + y2**2) * (x1 - x3)
        + (x3**2 + y3**2) * (x2 - x1)
    ) / D

    return (ux, uy)


def obtener_parametros_cilindro(grupos):
    parametros = []

    for grupo in grupos:
        if len(grupo) < 2:
            continue

        # Calcular el punto mínimo, máximo y medio
        punto_min = grupo[0]  # Primer punto del grupo
        punto_max = grupo[-1]  # Último punto del grupo
        indice_medio = len(grupo) // 2
        punto_medio = grupo[indice_medio]  # Punto medio

        # Calcular el centro del cilindro (promedio de los tres puntos)
        centro = centro_circulo(punto_min, punto_medio, punto_max)

        # Calcular el radio del cilindro
        radio = np.mean(
            [np.linalg.norm(centro - punto_min),
             np.linalg.norm(centro - punto_medio),
             np.linalg.norm(centro - punto_max)]
        )

        parametros.append(
            {
                "punto_min": punto_min,
                "punto_max": punto_max,
                "punto_medio": punto_medio,
                "radio": radio,
                "centro": centro,
            }
        )

    return parametros


def graficar_parametros_cilindro(grupos, parametros):
    """
    Grafica los puntos de cada grupo junto con los puntos mínimo, máximo y medio.

    Args:
        grupos: Lista de grupos de puntos (cada grupo es un array de puntos (x, y)).
        parametros: Lista de diccionarios con los parámetros de cada grupo (mínimo, máximo, medio).
    """
    plt.figure(figsize=(10, 8))

    # Colores para los grupos
    colores = plt.cm.get_cmap("tab10", len(grupos))

    for i, (grupo, parametro) in enumerate(zip(grupos, parametros)):
        # Graficar puntos del grupo
        plt.scatter(
            grupo[:, 0],
            grupo[:, 1],
            c=[colores(i)],
            s=5,
            alpha=0.7,
            label=f"Grupo {i + 1}",
        )

        # Graficar puntos mínimo, máximo y medio
        punto_min = parametro["punto_min"]
        punto_max = parametro["punto_max"]
        punto_medio = parametro["punto_medio"]

        plt.plot(
            punto_min[0],
            punto_min[1],
            "go",
            markersize=8,
            label="Min Grupo" if i == 0 else "",
        )
        plt.plot(
            punto_max[0],
            punto_max[1],
            "ro",
            markersize=8,
            label="Max Grupo" if i == 0 else "",
        )
        plt.plot(
            punto_medio[0],
            punto_medio[1],
            "yo",
            markersize=8,
            label="Medio Grupo" if i == 0 else "",
        )

        # Graficar el centro del cilindro
        centro = parametro["centro"]
        plt.plot(
            centro[0],
            centro[1],
            "bo",
            markersize=10,
            label="Centro Grupo" if i == 0 else "",
        )

    plt.title("Grupos de Puntos con Parámetros de Cilindros")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.show()


datos = parse_data("header.txt")
puntos, rangos_validos, angulos_validos = convertir_a_cartesiano(datos)

grupos_puntos = agrupar_puntos_por_distancia(puntos)

# Obtener parámetros de los cilindros
parametros_cilindros = obtener_parametros_cilindro(grupos_puntos)

# Graficar los grupos y sus parámetros
graficar_parametros_cilindro(grupos_puntos, parametros_cilindros)

# Escribimos los landmarks en un archivo
with open("landmarks.txt", "w") as f:
    for param in parametros_cilindros:
        f.write(f"{param['centro'][0]}, {param['centro'][1]}, {param['radio']}\n")
        