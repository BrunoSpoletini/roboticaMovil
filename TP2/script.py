import csv
import yaml
import numpy as np
import transforms3d.quaternions as quat
import matplotlib.pyplot as plt
from transforms3d import affines, euler, quaternions
import csv
import re 

inputFile = "log.txt"

def parseInput(input_file):
    output_file = "logParsed.csv"

    # regex para capturar sec y nanosec
    time_pattern = re.compile(r"sec=(\d+), nanosec=(\d+)")
 
    rows = []

    started = False

    with open(input_file, "r") as infile:
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

parseInput(inputFile)

poses = []
with open("./logParsed.csv", newline="") as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    for row in reader:
        if row[0].startswith("timestamp"):
            continue
        poses.append([float(x) for x in row[:8]])

def generarGrafico2D(x, y, x_label, y_label, tittle, pathfile, square=False, rad=False):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(
        x, y, color="blue"
    )
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

def generarGraficos():
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
        qs.append((round(q, 4) + 2*np.pi) % (2*np.pi))
        vs.append(round(v, 4))
        ws.append(round(w, 4))


    # Mostrar Path
    generarGrafico2D(xs, ys, "x", "y", "Camino", "path.png", square=True)

    # Mostrar Pose X
    generarGrafico2D(ts, xs, "t", "x", "Pose X", "pose_x.png")

    # Mostrar Pose Y
    generarGrafico2D(ts, ys, "t", "y", "Pose Y", "pose_y.png")
    
    # Mostrar Pose Orientacion
    generarGrafico2D(ts, np.cos(qs), "t", "θ", "Pose θ", "pose_q.png", rad=True)

    # Mostrar Velocidad Lineal
    generarGrafico2D(ts, vs, "t", "v", "Velocidad Lineal", "vel_lin.png")

    # Mostrar Velocidad Angular
    generarGrafico2D(ts, ws, "t", "w", "Velocidad Angular", "vel_ang.png")

generarGraficos()