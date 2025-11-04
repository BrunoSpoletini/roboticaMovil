#!/usr/bin/env python3

import numpy as np
import gtsam
from gtsam import Pose2

# Ejercicio 1
def parse_g20_file(filepath):
    poses = {}
    edges = []

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
                q11, q12, q13, q22, q23, q33 = map(float, tokens[6:12])

                # Generamos la matriz de información
                info = np.array([
                    [q11, q12, q13],
                    [q12, q22, q23],
                    [q13, q23, q33]
                ])

                # Obtenemos la covarianza
                cov = np.linalg.inv(info)

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
                continue


    return poses, edges

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
    if params is None:
        params = gtsam.GaussNewtonParams()
        params.setMaxIterations(100)
        params.setRelativeErrorTol(1e-6)

    # Creamos el optimizador con el grado, los valores iniciales y los parametros
    optimizer = gtsam.GaussNewtonOptimizer(graph, initial_poses, params)

    # Obtenemos el resultado de la optimizacion
    result = optimizer.optimize()

    return result, optimizer