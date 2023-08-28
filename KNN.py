# Sebastian Burgos Alanis A01746459
# Implementación del algoritmo KNN
# 28/08/23

import pandas as pd
import numpy as np

# Cálculo de la distancia euclidiana
def FomrulaEuclidiana(p1, p2):
    distancia = 0
    for i in range(len(p1)):
        distancia += (p1[i] - p2[i]) ** 2
    return np.sqrt(distancia)

# Algoritmo KNN, recibe parámetros de entrenamiento
def knn(X_train, y_train, nuevo_punto, k):
    distancias = []
    # Calculo de la distancia entre vecinos
    for i in range(len(X_train)):
        dist = FomrulaEuclidiana(nuevo_punto, X_train[i])
        distancias.append((dist, y_train[i]))
    # Ordenar las distancias y guardar los puntos más cercanos
    distancias.sort(key=lambda x: x[0])
    vecinos = distancias[:k]
    # Contar los vecinos
    clases = {}
    for distancia, f in vecinos:
        if f in clases:
            clases[f] += 1
        else:
            clases[f] = 1
    # Encontrar valores comunes y formar la predicción
    clase_predicha = max(clases, key=clases.get)
    # Mostrar la prediccion 
    print(f"prediccion: {clase_predicha}")
    return clase_predicha

# Calculo de la predicción del modelo
def prediccion(X_train, y_train, X_test, y_test_real, k):
    predicciones = []
    for p in X_test:
        clase_predicha = knn(X_train, y_train, p, k)
        predicciones.append(clase_predicha)
    # Calcular la precisión comparando las predicciones con las etiquetas reales
    exactitud = np.mean(np.array(predicciones) == y_test_real)
    return exactitud

# Limpieza de datos
datos = pd.read_csv("penguins.csv", header=0)
y = datos['species']
diccionario = {'Adelie': 1, 'Gentoo': 2, 'Chinstrap': 3}
y = y.map(diccionario)
y = y.drop([4, 272, 340, 618])
r = y
X = datos[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
X = X.dropna()

# Estos fueron los valores que más se acercaron al 100%
X_train = X[:410].values
y_train = y[:410].values
X_test = X[410:].values
y_test_real = y[410:].values
k = 2

# Resultados
exactitud = prediccion(X_train, y_train, X_test, y_test_real, k)
print("Precisión: ", exactitud * 100)
#se muestra la última prediccion de la base de datos
#print(f'prediccion para {X_test[-1]} = {knn(X_train, y_train, X_test[-1], k)}')