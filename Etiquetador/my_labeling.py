__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

import utl as utl

from utils_data import read_dataset, read_extended_dataset, crop_images, Plot3DCloud, visualize_k_means

import time
import matplotlib.pyplot as plt
import numpy as np
import Kmeans as km
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

def Kmean_statistics(kmeans_class, images, Kmax):
    wcds = []
    iterations = []
    times = []

    for k in range(2, Kmax+1):
        start_time = time.time()
        kmeans = kmeans_class(n_clusters=k, random_state=42)
        kmeans.fit(images)
        end_time = time.time()

        wcds.append(kmeans.inertia_)  # WCD
        iterations.append(kmeans.n_iter_)
        times.append(end_time - start_time)

    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.plot(range(2, Kmax+1), wcds, marker='o')
    plt.title('WCD by K')
    plt.xlabel('K')
    plt.ylabel('WCD')

    plt.subplot(132)
    plt.plot(range(2, Kmax+1), iterations, marker='o')
    plt.title('Iterations by K')
    plt.xlabel('K')
    plt.ylabel('Iterations')

    plt.subplot(133)
    plt.plot(range(2, Kmax+1), times, marker='o')
    plt.title('Time to Converge by K')
    plt.xlabel('K')
    plt.ylabel('Time (s)')

    plt.tight_layout()
    plt.show()

def Get_shape_accuracy(predicted_labels, true_labels):
    correct_matches = sum(p == t for p, t in zip(predicted_labels, true_labels))
    accuracy = (correct_matches / len(true_labels)) * 100
    return accuracy

def Get_color_accuracy(predicted_labels, true_labels):
    total_score = 0
    for pred, true in zip(predicted_labels, true_labels):
        intersection = set(pred).intersection(set(true))
        union = set(pred).union(set(true))
        total_score += len(intersection) / len(union) if union else 0

    accuracy = (total_score / len(true_labels)) * 100
    return accuracy

def retrieval_by_color(image_list, color_labels, query_colors, percentage=False):
    """
    Recibe una lista de etiquetas de color (cada etiqueta puede contener múltiples colores y sus respectivos porcentajes),
    y una lista de colores buscados. Retorna las imágenes cuyas etiquetas contienen los colores buscados, ordenadas
    opcionalmente por el porcentaje de coincidencia de color.

    :param image_list: Lista de imágenes.
    :param color_labels: Lista de etiquetas con colores y opcionalmente porcentajes.
    :param query_colors: Lista de colores a buscar.
    :param percentage: Booleano, si es True, se considera el porcentaje de color en el ordenamiento.
    :return: Lista de imágenes ordenadas por relevancia de coincidencia de color.
    """
    matches = []
    for index, colors in enumerate(color_labels):
        match = False
        total_percentage = 0
        for color, pct in colors.items():
            if color in query_colors:
                match = True
                if percentage:
                    total_percentage += pct

        if match:
            matches.append((total_percentage, index) if percentage else index)

    # Ordenamos por porcentaje si es requerido
    if percentage:
        matches.sort(reverse=True, key=lambda x: x[0])
        return [image_list[i[1]] for i in matches]
    else:
        return [image_list[i] for i in matches]

def retrieval_by_shape(image_list, shape_labels, query_shape, k_neighbors_percentage=False):
    """
    Recibe una lista de etiquetas de forma (cada etiqueta puede incluir un porcentaje de vecinos K con la misma forma),
    y una forma específica buscada. Retorna las imágenes cuyas etiquetas coinciden con la forma buscada, ordenadas
    opcionalmente por el porcentaje de vecinos K.

    :param image_list: Lista de imágenes.
    :param shape_labels: Lista de etiquetas con formas y opcionalmente porcentaje de vecinos K con la misma forma.
    :param query_shape: Forma específica a buscar.
    :param k_neighbors_percentage: Booleano, si es True, se considera el porcentaje de vecinos K en el ordenamiento.
    :return: Lista de imágenes ordenadas por relevancia de coincidencia de forma.
    """
    matches = []
    for index, (shape, pct) in enumerate(shape_labels):
        if shape == query_shape:
            matches.append((pct, index) if k_neighbors_percentage else index)

    # Ordenamos por porcentaje si es requerido
    if k_neighbors_percentage:
        matches.sort(reverse=True, key=lambda x: x[0])
        return [image_list[i[1]] for i in matches]
    else:
        return [image_list[i] for i in matches]

def retrieval_combined(image_list, color_labels, shape_labels, query_color, query_shape, use_percentage=True):
    """
    Función que combina la búsqueda por color y forma. Recibe listas de imágenes, etiquetas de color y forma,
    una consulta de color, una consulta de forma y un flag que indica si se debe utilizar el porcentaje para
    ordenar los resultados.

    :param image_list: Lista de imágenes.
    :param color_labels: Diccionario de etiquetas de color con porcentajes.
    :param shape_labels: Diccionario de etiquetas de forma con porcentajes de vecinos K.
    :param query_color: Lista de colores a buscar.
    :param query_shape: Forma específica a buscar.
    :param use_percentage: Si es True, ordena las imágenes por la suma de porcentajes de coincidencia de color y forma.
    :return: Lista de imágenes que coinciden con los criterios de búsqueda, ordenadas por relevancia.
    """
    color_matches = retrieval_by_color(image_list, color_labels, query_color, percentage=use_percentage)
    shape_matches = retrieval_by_shape(image_list, shape_labels, query_shape, k_neighbors_percentage=use_percentage)

    # Intersección de resultados, considerando los índices de las imágenes
    combined_indices = set([img[1] if use_percentage else img for img in color_matches]) & set([img[1] if use_percentage else img for img in shape_matches])

    # Extraer y ordenar las imágenes resultantes
    if use_percentage:
        # Extraemos pares (suma de porcentajes, índice) para las coincidencias
        final_matches = [(color_labels[idx][1] + shape_labels[idx][1], idx) for idx in combined_indices]
        final_matches.sort(reverse=True, key=lambda x: x[0])  # Ordenamos por la suma de porcentajes
        return [image_list[idx] for _, idx in final_matches]
    else:
        return [image_list[idx] for idx in combined_indices]

def calculate_accuracy(predictions, ground_truth):
    correct = np.sum(predictions == ground_truth)
    return correct / len(ground_truth) * 100


if __name__ == '__main__':
    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
    test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    # test_color_labels s'encarrega de cargar el dataset de colors per probar
    # train_imgs son totes les imatges que s'susaran per al entrenament del algosrime
    # train_class_labels donarà les labels per al entreno
    # train_color_labels donarà les labels per al entreno

    #print(train_imgs) # Carrega matrius de pixels possibles
    #print(test_imgs)  # Carrega matrius de pixels de cada foto disponible en ordre
    #print(train_class_labels) # Carrega totes les labels de classe disponibles
    #print(test_class_labels) # Carrega les labels de cada foto en ordre
    #print(train_color_labels) # Carrega totes els colors disponibles
    #print(test_color_labels) # Carrega els colors de cada foto en ordre

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    # You can start coding your functions here
    # Creem un clase KMeans, a la qual li fiquem la quantitat k de classes que volem amb opcions per defecte

    temp = []
    K_aux = km.KMeans(train_imgs[0])
    km.KMeans.find_bestK(K_aux, len(classes))
    km.KMeans.fit(K_aux)
    temp.append(km.get_colors(K_aux.centroids))
    print(temp)

    for color in utl.colors:
        index = []
        imagenes = []
        print("---------" + str(color) + "--------")
        i = 0
        for c in retrieval_by_color(temp, train_color_labels,color):
            imagenes.append((test_imgs[c]))
        Get_color_accuracy(temp, test_color_labels)

    ax = Plot3DCloud(K_aux)
    plt.show()

    color_accuracy = calculate_accuracy(kmeans_labels, test_color_labels)  # Simplified accuracy
    print(f"Color Clustering Accuracy: {color_accuracy:.2f}%")

    # KNN for shape classification
    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    knn_classifier.fit(train_imgs, train_class_labels)
    knn_predictions = knn_classifier.predict(test_imgs)
    shape_accuracy = calculate_accuracy(knn_predictions, test_class_labels)
    print(f"Shape Classification Accuracy: {shape_accuracy:.2f}%")



