# 🤖 Robótica Móvil – Trabajos Prácticos (UNR FCEIA)

Este repositorio contiene una serie de trabajos prácticos desarrollados para la materia **Robótica Móvil** (UNR – FCEIA), enfocados en cinemática, transformaciones, ROS2, visión por computadora y SLAM con grafos.
---

## 📁 Contenido del Repositorio

### 🧮 TP1 – Transformaciones

**Tema:** Transformaciones geométricas y cambios de referencia en robótica.

Se trabajan:
- Rotaciones intrínsecas y extrínsecas en 2D y 3D.
- Composición de transformaciones (rototraslaciones).
- Uso de matrices de rotación y coordenadas homogéneas.
- Análisis de ángulos de Euler y el problema de *Gimbal Lock*.
- Transformación de poses entre distintos sistemas de referencia (mundo, robot, cámara).
- Conversión de trayectorias del sistema IMU al sistema de la cámara usando matrices de calibración.

En este TP se construyen las bases matemáticas para manejar correctamente poses y cambios de frame en robótica.

---

### 🛞 TP2 – ROS2, Cinemática y Simulación

**Tema:** Cinemática de robots diferenciales, odometría y simulación en ROS2/Gazebo.

Se implementa y analiza:
- Cinemática de movimiento circular y diferencial.
- Relación entre velocidad lineal, angular y radio de giro.
- Cálculo de velocidades de ruedas para un TurtleBot3.
- Registro de odometría y velocidades desde ROS2.
- Scripts en Python para:
  - Parsear logs
  - Graficar trayectorias, orientación y velocidades
- Simulaciones en Gazebo con:
  - Trayectorias circulares y compuestas
  - Ejecución de secuencias de comandos `/cmd_vel`
- Procesamiento de datos de un láser:
  - Segmentación de puntos
  - Detección de cilindros
  - Estimación de centros y radios (landmarks)

Este TP conecta teoría de movimiento con experimentos reales en simulación.

---

### 👁️ TP3 – Visión por Computadora (Stereo)

**Tema:** Procesamiento de imágenes estéreo y reconstrucción 3D usando ROS2 y OpenCV.

Se desarrolla:
- Calibración de cámaras estéreo usando dataset EuRoC.
- Rectificación de imágenes con OpenCV.
- Extracción de features (FAST + BRIEF).
- Matching entre imágenes izquierda y derecha.
- Triangulación de puntos 3D a partir de correspondencias.
- Publicación de nube de puntos en ROS2 y visualización en RViz.
- Filtrado de matches espurios.
- Reconstrucción:
  - Dispersa (por matches)
  - Densa (usando mapa de disparidad)
- Visualización de:
  - Trayectorias de cámara
  - Nubes de puntos del entorno reconstruido

Este TP implementa un pipeline completo de visión estéreo y reconstrucción 3D.

---

### 🗺️ TP4 – Graph SLAM (2D y 3D) con GTSAM

**Tema:** SLAM basado en grafos de factores, en 2D y 3D.

Se implementa:

#### 🔹 Graph SLAM 2D
- Construcción de factor graph con:
  - Poses como nodos
  - Restricciones relativas como aristas
- Optimización batch con **Gauss-Newton**.
- Análisis de mínimos locales y necesidad de perturbar las poses iniciales.
- Optimización incremental usando **iSAM2**.
- Comparación entre:
  - Trayectorias iniciales
  - Trayectorias optimizadas

#### 🔹 Graph SLAM 3D
- Extensión del mismo enfoque a poses 3D con cuaterniones.
- Construcción del grafo con `Pose3` y factores relativos.
- Optimización:
  - Batch con Gauss-Newton
  - Incremental con iSAM2
- Visualización de trayectorias en 2D y 3D.

Este TP muestra un pipeline completo de SLAM moderno basado en optimización de grafos.

---

## 🛠️ Tecnologías y Herramientas

- Python
- ROS2
- OpenCV
- GTSAM
- NumPy / Matplotlib
- Gazebo / RViz
- Datasets: EuRoC MAV

---

## 🎯 Objetivo del Repositorio

Reunir implementaciones prácticas de conceptos fundamentales de robótica móvil:

- Cinemática y transformaciones
- Percepción con visión estéreo
- Reconstrucción 3D
- SLAM basado en grafos (2D y 3D)
- Integración con ROS2 y herramientas de simulación
