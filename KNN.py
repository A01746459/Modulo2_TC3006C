import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

"""Cálculo fórmula Euclidiana"""
#Se utiliza para saber la distancia entre los puntos 
def FormulaEuclidiana(p1, p2):
    distancia = 0
    # Calcular la suma de las diferencias al cuadrado de cada dimensión
    for i in range(len(p1)):
        distancia += (p1[i] - p2[i]) ** 2
    # Calcular la raíz cuadrada de la suma para obtener la distancia euclidiana
    return np.sqrt(distancia)

"""Algoritmo K-Nearest Neighbors (KNN)"""
#Funcion KNN donde recibe valores de entrenamiento, prueba, punto y valor de k
def knn(X_train, y_train, nuevo_punto, k):
    distancias = []
    
    # Calcular la distancia entre el nuevo punto y los puntos de entrenamiento
    for i in range(len(X_train)):
        dist = FormulaEuclidiana(nuevo_punto, X_train[i])
        distancias.append((dist, y_train[i]))
    
    # Ordenar las distancias y seleccionar los k vecinos más cercanos
    distancias.sort(key=lambda x: x[0])
    vecinos = distancias[:k]
    
    # Contar las ocurrencias de cada clase entre los vecinos
    clases = {}
    for distancia, f in vecinos:
        if f in clases:
            clases[f] += 1
        else:
            clases[f] = 1
    
    # Elegir la clase más común entre los vecinos como predicción
    clase_predicha = max(clases, key=clases.get)
    return clase_predicha

"""Función para calcular las predicciones en un conjunto de prueba"""
def prediccion(X_train, y_train, X_test, y_test_real, k):
    predicciones = []
    for i, p in enumerate(X_test):
        clase_predicha = knn(X_train, y_train, p, k)
        predicciones.append(clase_predicha)
        # Imprimir el valor real y el valor predicho para cada punto de prueba
        print(f'Valor de prueba: {y_test_real[i]}, Valor predicho: {clase_predicha}')
    return predicciones

"""Función para calcular una matriz de confusión detallada"""
def matriz_confusion_detallada(y_real, y_pred):
    clases_reales = np.unique(y_real)
    n_clases = len(clases_reales)
    matriz = np.zeros((n_clases, n_clases))
    for i in range(n_clases):
        for j in range(n_clases):
            # Calcular la cantidad de instancias donde el valor real y el valor predicho coinciden
            matriz[i][j] = np.sum((y_real == clases_reales[i]) & (y_pred == clases_reales[j]))
    return matriz

"""Carga de datos y preprocesamiento"""
datos = pd.read_csv("penguins.csv", header=0)
y = datos['species']
# Mapear las etiquetas de clase a valores numéricos
diccionario = {'Adelie': 1, 'Gentoo': 2, 'Chinstrap': 3}
y = y.map(diccionario)
# Eliminar algunas instancias específicas del conjunto de datos
y = y.drop([4, 272, 340, 618])
r = y
X = datos[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
# Eliminar filas con valores faltantes
X = X.dropna()

# División de datos en conjuntos de entrenamiento y prueba
X_train = X[:510].values
y_train = y[:510].values
X_test = X[510:].values
y_test_real = y[510:].values
k = 2

# Realizar predicciones
predicciones = prediccion(X_train, y_train, X_test, y_test_real, k)

# Calcular y mostrar la matriz de confusión detallada
matriz_confusion = matriz_confusion_detallada(y_test_real, predicciones)
print("\nMatriz de Confusión Detallada:")
print(matriz_confusion)

# Calcular la precisión
exactitud = np.mean(np.array(predicciones) == y_test_real)
print("\nPrecisión: \n", exactitud * 100, "\n")

# Mostrar la precision, recall y f1-score
classification_report_str = classification_report(y_test_real, predicciones)
print("\n clases:\n ",classification_report_str)