import csv
import yaml
import numpy as np
import transforms3d.quaternions as quat
import matplotlib.pyplot as plt
from transforms3d import affines, euler, quaternions

# Leer matriz T_BS de sensor.yaml
with open("./mav0/cam0/sensor.yaml", "r") as f:
    sensor = yaml.safe_load(f)
T_BS = np.array(sensor["T_BS"]["data"]).reshape((4, 4))
T_SB = np.linalg.inv(T_BS)  # de sensor a body

poses = []
with open("./mav0/state_groundtruth_estimate0/data.csv", newline="") as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    for row in reader:
        if row[0].startswith("#"):
            continue
        poses.append([float(x) for x in row[:8]])


def generarGrafico():
    # Guardar trayectorias para graficar
    imu_traj = []  # ground-truth original (IMU)
    cam_traj = []  # ground-truth transformado (cámara)

    for pose in poses:
        timestamp, x, y, z, qw, qx, qy, qz = pose
        # Trayectoria IMU (sistema de coordenadas de la IMU/body)
        imu_traj.append([x, y, z])
        # Pose del robot en sistema de coordenadas de la IMU/body
        p_B = np.array([x, y, z, 1.0])
        q_B = np.array([qw, qx, qy, qz])
        R_B = quat.quat2mat(q_B)

        # Aplicar rotación del body a la posición de la cámara
        p_C = R_B @ T_SB[:3, 3] + np.array([x, y, z])
        cam_traj.append([p_C[0], p_C[1], p_C[2]])

    # Graficar ambos caminos en el sistema de coordenadas del ground-truth original
    imu_traj = np.array(imu_traj)
    cam_traj = np.array(cam_traj)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        imu_traj[:, 0], imu_traj[:, 1], imu_traj[:, 2], label="IMU (Body)", color="blue"
    )
    ax.plot(
        cam_traj[:, 0],
        cam_traj[:, 1],
        cam_traj[:, 2],
        label="Cámara Izquierda",
        color="red",
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Trayectorias Ground-Truth")
    ax.legend()
    plt.tight_layout()
    plt.savefig("trayectorias.png", dpi=300)


def make_T(R, t):
    return affines.compose(t, R, [1, 1, 1])


def mostrarDatos():
    output = []
    for pose in poses:
        timestamp, x, y, z, qw, qx, qy, qz = pose

        # Pose del robot en sistema de coordenadas de la IMU/body
        p_B = np.array([x, y, z, 1.0])
        q_B = np.array([qw, qx, qy, qz])
        R_B = quat.quat2mat(q_B)

        Bi_T_B0 = make_T(R_B, p_B[:3])

        # Rototraslacion del sistema de coordenadas body a cámara
        # S_T_B0

        S_T_B0 = T_SB @ Bi_T_B0
        # Convertimos a desplazamiento y cuaternion
        p_S = S_T_B0[:3, 3]
        R_S = S_T_B0[:3, :3]
        q_S = quat.mat2quat(R_S)
        qw, qx, qy, qz = q_S

        timestamp_s = float(timestamp) / 1e9
        output.append(
            [
                f"{timestamp_s:.9f}",
                f"{p_S[0]:.4f}",
                f"{p_S[1]:.4f}",
                f"{p_S[2]:.4f}",
                f"{qw:.4f}",
                f"{qx:.4f}",
                f"{qy:.4f}",
                f"{qz:.4f}",
            ]
        )

    # Guardar en salida.csv
    with open("salida.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "x", "y", "z", "qw", "qx", "qy", "qz"])
        for row in output:
            writer.writerow(row)


mostrarDatos()
