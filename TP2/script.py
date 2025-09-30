import csv
import yaml
import numpy as np
import transforms3d.quaternions as quat
import matplotlib.pyplot as plt
from transforms3d import affines, euler, quaternions
import csv
import re
import os 

def parseInput(input_file):
    output_file = f"{input_file}_parsed.csv"

    # regex para capturar sec y nanosec
    time_pattern = re.compile(r"sec=(\d+), nanosec=(\d+)")
 
    rows = []

    started = False

    with open(f"{input_file}.txt", "r") as infile:
        for line in infile:

            time_match = time_pattern.search(line)
            if not time_match:
                continue

            sec, nanosec = time_match.groups()
            # generar timestamp como sec.nanosec
            timestamp = float(sec) + (float(nanosec) * 1e-9)

            # extraer los valores flotantes después del tabulador
            parts = line.split("\t")[1:]
            floats = [p.strip() for p in parts if p.strip()]

            if(float(floats[1]) > 0.001 and float(floats[2]) > 0.001):
                started = True

            # fila con timestamp + valores
            if(started):
                row = [timestamp] + floats
                rows.append(row)


    # armar encabezados dinámicos según cantidad de columnas
    if rows:
        header = ["timestamp", "x", "y", "θ", "v", "w"]

        with open(output_file, "w", newline="") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(header)
            writer.writerows(rows)

def readPoses(input_file):
    poses = []
    with open(f"./{input_file}_parsed.csv", newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            if row[0].startswith("timestamp"):
                continue
            poses.append([float(x) for x in row[:8]])
    return poses

def obtenerMaxMin(xs, ys, qs, vs, ws):
    max_x = max(xs)
    min_x = min(xs)
    print("Rango X: ", min_x, " a ", max_x)
    max_y = max(ys)
    min_y = min(ys)
    print("Rango Y: ", min_y, " a ", max_y)
    max_q = max(qs)
    min_q = min(qs)
    print("Rango θ: ", min_q, " a ", max_q)
    max_v = max(vs)
    min_v = min(vs)
    print("Rango Vel. linear: ", min_v, " a ", max_v)
    max_w = max(ws)
    min_w = min(ws)
    print("Rango Vel. angular: ", min_w, " a ", max_w)

    return (max_x, min_x, max_y, min_y, max_q, min_q, max_v, min_v, max_w, min_w)

def generarGrafico2D(x, y, x_label, y_label, tittle, pathfile, square=False, rad=False, dots=False, orientations=None):

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(
        x, y, color="blue", label=tittle
    )
    if(dots):
        n = int(len(x) / 4)
        indices = [n, 2*n, 3*n]
        ax.scatter([x[i] for i in indices], [y[i] for i in indices],s=100, color="red", alpha=1)
        ax.scatter([x[0]], [y[0]], s=100, marker="s", color="green", label="Inicio", alpha=1)
        ax.scatter([x[-1]], [y[-1]], s=100, marker="^", color="green", label="Fin", alpha=1)
        
        if orientations is not None:
            arrow_indices = indices + [0, len(x)-1]
            for i in arrow_indices:
                if i < len(orientations):
                    dx = 0.3 * np.cos(orientations[i])
                    dy = 0.3 * np.sin(orientations[i])
                    ax.arrow(x[i], y[i], dx, dy, head_width=0.1, head_length=0.1, 
                            fc='orange', ec='orange', alpha=1, width=0.01)



    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(tittle)
    ax.legend()
    if (square):
        ax.set_aspect("equal", adjustable="box")
    if (rad):
        ax.set_ylim(-np.pi, np.pi)
        ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])

    plt.tight_layout()
    plt.savefig(pathfile, dpi=300)

def generarGraficos(inputFile, outputDir, dots=False):
    os.makedirs(outputDir, exist_ok=True)
    
    parseInput(inputFile) # Genera csv parseado
    poses = readPoses(inputFile) # Lee poses del csv

    ts = []
    xs = []
    ys = []  
    qs = []
    vs = []
    ws = []

    for data in poses:
        timestamp, x, y, q, v, w = data
        ts.append(timestamp)
        xs.append(round(x, 4))
        ys.append(round(y, 4))
        qs.append(round(q, 4))
        vs.append(round(v, 4))
        ws.append(round(w, 4))

    # Mostrar Path
    generarGrafico2D(xs, ys, "x", "y", "Camino", f"./{outputDir}/path.png", square=True, dots=dots, orientations=qs)

    # Mostrar Pose X
    generarGrafico2D(ts, xs, "t", "x", "Trayectoria X", f"./{outputDir}/pose_x.png", dots=dots)

    # Mostrar Pose Y
    generarGrafico2D(ts, ys, "t", "y", "Trayectoria Y", f"./{outputDir}/pose_y.png", dots=dots)

    # Mostrar Pose Orientacion
    generarGrafico2D(ts, qs, "t", "θ", "Orientación θ", f"./{outputDir}/pose_q.png", rad=True, dots=dots)

    # Mostrar Velocidad Lineal
    generarGrafico2D(ts, vs, "t", "v", "Velocidad Lineal", f"./{outputDir}/vel_lin.png")

    # Mostrar Velocidad Angular
    generarGrafico2D(ts, ws, "t", "w", "Velocidad Angular", f"./{outputDir}/vel_ang.png")

    # Mostrar valores maximos y minimos
    obtenerMaxMin(xs, ys, qs, vs, ws)

# generarGraficos("log", "ej5")

generarGraficos("log", "ej6", dots=True)
