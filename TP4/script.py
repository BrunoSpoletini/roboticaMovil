#!/usr/bin/env python3

import numpy as np
import gtsam
from gtsam import Pose2, Pose3, Rot3
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Apartado 2.1 A

# Parsea un archivo g2o para obtener las poses y aristas del dataset 2D
def parse_g2o_file(filepath):

    poses = {}
    edges = []

    # Leemos el archivo donde esta el dataset
    with open(filepath, 'r') as f:
        for row in f:
            
            # Chequeamos que no sea una fila vacia
            if not row.strip():
                continue

            # Separamos la fila en elementos
            tokens = row.strip().split()

            # Verificamos si estamos con una pose
            if tokens[0] == "VERTEX_SE2":

                # Obtenemos los datos
                i = int(tokens[1])
                x, y, theta = map(float, tokens[2:5])

                # Guardamos la pose
                poses[i] = (x, y, theta)

            # Verificamos si estamos con una arista
            elif tokens[0] == "EDGE_SE2":

                # Obtenemos los datos
                i = int(tokens[1])
                j = int(tokens[2])
                x, y, theta = map(float, tokens[3:6])

                # Extraemos el vector de información
                q11, q12, q13, q22, q23, q33 = map(float, tokens[6:12])

                # Reconstruimos la matriz de información 6x6 desde el vector triangular superior
                info = np.array([
                    [q11, q12, q13],
                    [q12, q22, q23],
                    [q13, q23, q33]
                ])

                # Calculamos la matriz de covarianza 
                try:
                    cov = np.linalg.inv(info)
                except np.linalg.LinAlgError:
                    # Si la matriz no es invertible, usamos pseudo-inversa
                    cov = np.linalg.pinv(info)

                # Guardamos la arista
                edges.append({
                    'i': i,
                    'j': j,
                    'x': x,
                    'y': y,
                    'theta': theta,
                    'info': info,
                    'cov': cov
                })
            
            else:
                # Ignoramos otros tipos de líneas
                continue


    return poses, edges

# Apartado 3.1 A

# Parsea un archivo g2o para obtener las poses y aristas del dataset 3D
def parse_g2o_3d_file(filepath):

    poses = {}
    edges = []
    
    # Leemos el archivo donde esta el dataset
    with open(filepath, 'r') as f:
        for line in f:

            # Chequeamos que no sea una fila vacia
            if not line.strip():
                continue
            
            # Separamos la línea en elementos
            tokens = line.strip().split()
            
            # Verificamos si es una pose 3D
            if tokens[0] == "VERTEX_SE3:QUAT":

                # Obtenemos los datos
                i = int(tokens[1])
                x, y, z = map(float, tokens[2:5])
                qx, qy, qz, qw = map(float, tokens[5:9])
                
                # Guardamos la pose
                poses[i] = (x, y, z, qx, qy, qz, qw)
            
            # Verificamos si es una arista 3D
            elif tokens[0] == "EDGE_SE3:QUAT":

                # Obtenemos los datos
                i = int(tokens[1])
                j = int(tokens[2])
                x, y, z = map(float, tokens[3:6])
                qx, qy, qz, qw = map(float, tokens[6:10])
                
                # Extraemos el vector de información
                info_vector = list(map(float, tokens[10:31]))
                
                # Reconstruimos la matriz de información 6x6 desde el vector triangular superior
                info = np.zeros((6, 6))
                idx = 0
                for row in range(6):
                    for col in range(row, 6):
                        info[row, col] = info_vector[idx]
                        info[col, row] = info_vector[idx]  # Por simetría
                        idx += 1
                
                # Calculamos la matriz de covarianza 
                try:
                    cov = np.linalg.inv(info)
                except np.linalg.LinAlgError:
                    # Si la matriz no es invertible, usamos pseudo-inversa
                    cov = np.linalg.pinv(info)
                
                # Guardamos la arista
                edges.append({
                    'i': i,
                    'j': j,
                    'x': x,
                    'y': y,
                    'z': z,
                    'qx': qx,
                    'qy': qy,
                    'qz': qz,
                    'qw': qw,
                    'info': info,
                    'cov': cov
                })
            
            else:
                # Ignoramos otros tipos de líneas
                continue
    
    return poses, edges

# Apartado 2.2 B - Batch solution

# Genera un NonLinearFactorGraph con las poses y aristas pasadas como argumento
def generate_graph(poses, edges):

    # Creamos el grafo y los valores iniciales
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    # Insertamos las poses iniciales
    for idx, (x, y, th) in poses.items():
        initial.insert(idx, Pose2(x, y, th))

    # Agregamos las aristas
    for edge in edges:

        # Creamos una Pose2 de la medición relativa
        meas = Pose2(edge['x'], edge['y'], edge['theta'])
        
        # Obtenemos el ruido a partir de la covarianza
        noise = gtsam.noiseModel.Gaussian.Covariance(edge['cov'])

        # Añadir un BetweenFactorPose2 al grafo con la informacion
        factor = gtsam.BetweenFactorPose2(edge['i'], edge['j'], meas, noise)
        graph.add(factor)

    # Anclamos la primer pose para evitar soluciones flotantes
    if len(poses) > 0:

        # Obtenemos la primer pose
        i0 = min(poses.keys())
        x0, y0, th0 = poses[i0]
        pose0 = Pose2(x0, y0, th0)

        # Definimos una covarianza pequena para la primer pose
        cov0 = np.diag([1e-6, 1e-6, 1e-8])
        noise0 = gtsam.noiseModel.Gaussian.Covariance(cov0)
        
        # Agregamos la primer pose al grafo
        graph.add(gtsam.PriorFactorPose2(i0, pose0, noise0))

    return graph, initial

# Perturba los valores pasados por argumento
def pertubateValues(old_poses, sigma, seed):

    new_poses = gtsam.Values()

    np.random.seed(seed)

    # Perturbamos cada una de las poses
    for i in old_poses.keys():

        # Obtenemos los valores de la pose
        pi = old_poses.atPose2(i)
        xi = pi.x() + np.random.normal(0, sigma[0])
        yi = pi.y() + np.random.normal(0, sigma[1])
        thetai = pi.theta() + np.random.normal(0, sigma[2])

        # Creamos la nueva pose perturbada
        new_poses.insert(i, Pose2(xi, yi, thetai))

    return new_poses

# Optimiza el grafo y los valores iniciales por medio de gauss newton
def OptimizeGaussNewton(graph, initial_poses):

    # Definimos los parametros del Gauss Newton
    params = gtsam.GaussNewtonParams()
    params.setMaxIterations(100)
    params.setRelativeErrorTol(1e-6)

    # Creamos el optimizador con el grado, los valores iniciales y los parametros
    optimizer = gtsam.GaussNewtonOptimizer(graph, initial_poses, params)

    # Obtenemos el resultado de la optimizacion
    result = optimizer.optimize()

    return result

# Extrae los puntos de la trayectoria de poses 2D
def extract_trayectory(poses):

    indexes = list(poses.keys())

    # Filtramos solo keys que son enteros
    filtered_indexes = [k for k in indexes if isinstance(k, int) or (hasattr(k, 'toInt') and isinstance(k, int))]
    
    # Ordenamos los indices
    sorted_indexes = sorted(filtered_indexes)

    # Extramos los (x, y) de las poses
    xs, ys = [], []

    for index in sorted_indexes:

        p = poses.atPose2(index)
        xs.append(p.x()); ys.append(p.y())

    return np.array(xs), np.array(ys)

# Plotea las poses inciales y optimizados
def plot_poses(initial_poses, optimized_poses, outputdir, title, filename, plot=False):

    # Extramos los puntos de la trayectoria de los poses iniciales y optimizados
    ixs, iys = extract_trayectory(initial_poses)
    oxs, oys = extract_trayectory(optimized_poses)

    # Generamos una figura con las poses iniciales y las poses optimizadas
    plt.figure(figsize=(8,6))
    plt.plot(ixs, iys, 'o-', label='Poses Iniciales', markersize=2)
    plt.plot(oxs, oys, 'o-', label='Poses Optimizados', markersize=2)
    plt.axis('equal')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(title)
    plt.legend()
    plt.savefig(f'{outputdir}/{filename}.png', dpi=300)

    if plot:
        plt.show()
    else:
        plt.close()

# Apartado 3.2 B 

# Genera un NonLinearFactorGraph con las poses y aristas pasadas como argumento
def generate_graph_3d(poses, edges):

    # Creamos el grafo y los valores iniciales
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    
    # Insertamos las poses iniciales
    for idx, (x, y, z, qx, qy, qz, qw) in poses.items():

        # Creamos la rotación desde el cuaternión
        rotation = Rot3.Quaternion(qw, qx, qy, qz)

        # Creamos la traslación
        translation = np.array([x, y, z])
        
        # Creamos la Pose3
        pose = Pose3(rotation, translation)

        # Insertamos en las poses iniciales
        initial.insert(idx, pose)
    
    # Agregamos las aristas
    for edge in edges:

        # Creamos la rotación relativa desde el cuaternión
        rotation = Rot3.Quaternion(edge['qw'], edge['qx'], edge['qy'], edge['qz'])

        # Creamos la traslación relativa
        translation = np.array([edge['x'], edge['y'], edge['z']])

        # Creamos la Pose3 de la medición relativa
        meas = Pose3(rotation, translation)
        
        # Obtenemos el modelo de ruido a partir de la covarianza
        noise = gtsam.noiseModel.Gaussian.Covariance(edge['cov'])
        
        # Añadir un BetweenFactorPose2 al grafo con la informacion
        factor = gtsam.BetweenFactorPose3(edge['i'], edge['j'], meas, noise)
        graph.add(factor)
        
    
    # Anclamos la primera pose para evitar soluciones flotantes
    if len(poses) > 0:

        # Obtenemos la primera pose
        i0 = min(poses.keys())
        x0, y0, z0, qx0, qy0, qz0, qw0 = poses[i0]
        
        # Creamos la Pose3 inicial
        rotation0 = Rot3.Quaternion(qw0, qx0, qy0, qz0)
        translation0 = np.array([x0, y0, z0])
        pose0 = Pose3(rotation0, translation0)
        
        # Definimos una covarianza pequeña para la primera pose
        cov0 = np.diag([1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-8])
        noise0 = gtsam.noiseModel.Gaussian.Covariance(cov0)

        # Agregamos la primera pose al grafo
        graph.add(gtsam.PriorFactorPose3(i0, pose0, noise0))
    
    return graph, initial

# Extrae los puntos de la trayectoria de poses 3D
def extract_trayectory_3d(poses):

    indexes = list(poses.keys())
    
    # Filtramos solo keys que son enteros
    filtered_indexes = [k for k in indexes if isinstance(k, int) or (hasattr(k, 'toInt') and isinstance(k, int))]
    
    # Ordenamos los índices
    sorted_indexes = sorted(filtered_indexes)
    
    # Extraemos las coordenadas (x, y, z) de las poses
    xs, ys, zs = [], [], []
    
    for index in sorted_indexes:
        p = poses.atPose3(index)
        t = p.translation()
        xs.append(t[0])
        ys.append(t[1])
        zs.append(t[2])
    
    return np.array(xs), np.array(ys), np.array(zs), sorted_indexes

# Ploteamos las poses 3D (vista superior x-y y vista 3D)
def plot_poses_3d(initial_poses, optimized_poses, outputdir, filename, plot=False):

    # Extraemos los puntos de las trayectorias
    ixs, iys, izs, _ = extract_trayectory_3d(initial_poses)
    oxs, oys, ozs, _ = extract_trayectory_3d(optimized_poses)
    
    # Creamos la figura con dos subplots
    fig = plt.figure(figsize=(16, 6))
    
    # Subplot 1: Vista superior (x-y)
    ax1 = fig.add_subplot(121)
    ax1.plot(ixs, iys, 'o-', label='Poses Iniciales', markersize=2, alpha=0.7)
    ax1.plot(oxs, oys, 'o-', label='Poses Optimizadas', markersize=2, alpha=0.7)
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title('Vista Superior (x-y)')
    ax1.legend()
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Vista 3D
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(ixs, iys, izs , 'o-', label='Poses Iniciales', markersize=2, alpha=0.7)
    ax2.plot(oxs, oys, ozs , 'o-', label='Poses Optimizadas', markersize=2, alpha=0.7)
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.set_zlabel('z (m)')
    ax2.set_title('Vista 3D')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{outputdir}/{filename}.png', dpi=300)

    if plot:
        plt.show()
    else:
        plt.close()

# Apartado 2.3 C - Incremental solution

# Calcula una solucion incremental con ISAM2
def incremental_solution(poses, edges):

    # Inicializamos el solver ISAM2
    isam = gtsam.ISAM2()
    result = None

    # Iteramos por cada pose
    for idx, (x, y, th) in poses.items():
        
        # Inicializamos el grafo de factores y la estimación inicial
        graph = gtsam.NonlinearFactorGraph()
        initialEstimate = gtsam.Values()
        
        if idx == 0: # Primera pose
            priorNoise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6, 1e-6, 1e-8]))
            graph.add(gtsam.PriorFactorPose2(0, Pose2(x, y, th), priorNoise))
            initialEstimate.insert(idx, Pose2(x, y, th))
        else:
            # No es la primera pose: usamos la última pose optimizada como initial estimate
            prevPose = result.atPose2(idx - 1)
            initialEstimate.insert(idx, prevPose)
        
        # Iteramos sobre cada arista
        for edge in edges:
            ide1 = edge['i']
            ide2 = edge['j']
            dx = edge['x']
            dy = edge['y']
            dtheta = edge['theta']
            
            # Si la arista termina en la pose actual
            if ide2 == idx:
                # Construimos el modelo de ruido a partir de la covarianza
                cov = edge['cov']
                Model = gtsam.noiseModel.Gaussian.Covariance(cov)
                
                # Agregamos el BetweenFactor al grafo
                graph.add(gtsam.BetweenFactorPose2(ide1, ide2, Pose2(dx, dy, dtheta), Model))
        
        # Actualizamos ISAM2 con el grafo y la estimación inicial
        isam.update(graph, initialEstimate)
        
        # Calculamos la estimación actual
        result = isam.calculateEstimate()
    
    return result


# Apartado 3.3 C - Incremental solution 3D

# Calcula una solucion incremental con ISAM2
def incremental_solution_3d(poses, edges):

    # Inicializamos el solver ISAM2
    isam = gtsam.ISAM2()
    result = None
    
    # Iteramos por cada pose
    for idx, (x, y, z, qx, qy, qz, qw) in poses.items():
        
        # Inicializamos el grafo de factores y la estimación inicial para este paso
        graph = gtsam.NonlinearFactorGraph()
        initialEstimate = gtsam.Values()
        
        if idx == 0:  # Primera pose
            # Creamos la primera pose
            rotation0 = Rot3.Quaternion(qw, qx, qy, qz)
            translation0 = np.array([x, y, z])
            pose0 = Pose3(rotation0, translation0)
            
            # Agregamos un prior factor para anclar el sistema de referencia
            priorNoise = gtsam.noiseModel.Diagonal.Sigmas(
                np.array([1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-8])
            )
            graph.add(gtsam.PriorFactorPose3(idx, pose0, priorNoise))
            initialEstimate.insert(idx, pose0)
        else:
            # No es la primera pose: usamos la pose del archivo como estimación inicial
            rotation = Rot3.Quaternion(qw, qx, qy, qz)
            translation = np.array([x, y, z])
            pose = Pose3(rotation, translation)
            initialEstimate.insert(idx, pose)
        
        # Buscamos todas las aristas que terminan en la pose actual
        for edge in edges:
            ide1 = edge['i']
            ide2 = edge['j']
            
            # Si la arista termina en la pose actual
            if ide2 == idx:
                # Creamos la transformación relativa (medición)
                rotation_rel = Rot3.Quaternion(edge['qw'], edge['qx'], edge['qy'], edge['qz'])
                translation_rel = np.array([edge['x'], edge['y'], edge['z']])
                meas = Pose3(rotation_rel, translation_rel)
                
                # Construimos el modelo de ruido desde la covarianza 6x6
                cov = edge['cov']
                noiseModel = gtsam.noiseModel.Gaussian.Covariance(cov)
                
                # Agregamos el BetweenFactor al grafo
                graph.add(gtsam.BetweenFactorPose3(ide1, ide2, meas, noiseModel))
        
        # Actualizamos ISAM2 incrementalmente
        isam.update(graph, initialEstimate)
        
        # Calculamos la mejor estimación actual
        result = isam.calculateEstimate()
    
    return result

# Funcion principal del script
def main(g20_pathfile, g2o3d_pathfile, outputdir, mode):

    # Creamos el directorio de salida si el mismo no existe
    os.makedirs(outputdir, exist_ok=True)

    if mode == '2d' or mode == 'both':
        print("=== Ejecutando optimización 2D ===")

        # --- Ejercicio 2 - Apartado A ---
        poses, edges = parse_g2o_file(g20_pathfile)

        # --- Ejercicio 2 - Apartado B ---
        graph, initial_poses = generate_graph(poses, edges) 
        perturbed_initial_poses = pertubateValues(initial_poses, (0.2, 0.2, 0.1), 0)

        # Optimización batch con Gauss-Newton
        optimized_poses = OptimizeGaussNewton(graph, initial_poses)
        plot_poses(initial_poses, optimized_poses, outputdir, 'Poses: inicial vs optimizada (Gauss-Newton)', "posesIniciales_batch")

        # Optimización batch con Gauss-Newton con poses perturbadas
        optimized_perturbed_poses = OptimizeGaussNewton(graph, perturbed_initial_poses)
        plot_poses(perturbed_initial_poses, optimized_perturbed_poses, outputdir, 'Poses perturbadas: inicial vs optimizada (Gauss-Newton)', "posesPerturbadas_batch")

        # --- Ejercicio 2 - Apartado C ---
        # Optimización incremental con ISAM2
        optimized_poses_isam = incremental_solution(poses, edges)
        plot_poses(initial_poses, optimized_poses_isam, outputdir, 'Poses: inicial vs optimizada (ISAM2)', "poses_incremental_isam2")

        print(f"Trayectorias optimizadas 2D guardadas en el directorio {outputdir}.")

    if mode == '3d' or mode == 'both':
        print("\n=== Ejecutando optimización 3D ===")

        # --- Ejercicio 3 - Apartado A ---
        poses_3d, edges_3d = parse_g2o_3d_file(g2o3d_pathfile)
        
        # --- Ejercicio 3 - Apartado B ---
        graph_3d, initial_poses_3d = generate_graph_3d(poses_3d, edges_3d)

        # Optimización batch 3D
        optimized_poses_3d = OptimizeGaussNewton(graph_3d, initial_poses_3d)        
        plot_poses_3d(initial_poses_3d, optimized_poses_3d, outputdir, "poses_3d_garage_batch")

        # --- Ejercicio 3 - Apartado C ---
        # Optimización incremental con ISAM2
        optimized_poses_3d_isam = incremental_solution_3d(poses_3d, edges_3d)
        plot_poses_3d(initial_poses_3d, optimized_poses_3d_isam, outputdir, "poses_3d_garage_incremental_isam2")

        print(f"Trayectorias optimizadas 3D guardadas en el directorio {outputdir}.")

if __name__ == '__main__':

    # Parseamos los argumentos
    parser = argparse.ArgumentParser(description="Batch SLAM con GTSAM (2D y 3D)")
    parser.add_argument('--g2o', type=str, default='input_INTEL_g2o.g2o', help='ruta al archivo .g2o 2D')
    parser.add_argument('--g2o3d', type=str, default='parking-garage.g2o', help='ruta al archivo .g2o 3D')
    parser.add_argument('--output', type=str, default='output', help='Directorio de salida')
    parser.add_argument('--mode', type=str, default='both', choices=['2d', '3d', 'both'], 
                        help='Modo de operación: 2d, 3d o both')
    args = parser.parse_args()

    # Ejecutamos el script
    main(args.g2o, args.g2o3d, args.output, args.mode)
