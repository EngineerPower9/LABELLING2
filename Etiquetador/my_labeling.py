__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

import utl as utl

from Etiquetador.KNN import KNN
from Etiquetador.utils import rgb2gray
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

    for k in range(2, Kmax + 1):
        start_time = time.time()
        kmeans = kmeans_class(n_clusters=k, random_state=42)
        kmeans.fit(images)
        end_time = time.time()

        wcds.append(kmeans.inertia_)  # WCD
        iterations.append(kmeans.n_iter_)
        times.append(end_time - start_time)

    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.plot(range(2, Kmax + 1), wcds, marker='o')
    plt.title('WCD by K')
    plt.xlabel('K')
    plt.ylabel('WCD')

    plt.subplot(132)
    plt.plot(range(2, Kmax + 1), iterations, marker='o')
    plt.title('Iterations by K')
    plt.xlabel('K')
    plt.ylabel('Iterations')

    plt.subplot(133)
    plt.plot(range(2, Kmax + 1), times, marker='o')
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
    images_return = []
    porcentajes_list = []
    for i, image in enumerate(image_list):
        if query_colors in color_labels[i]:
            # CONTAMOS VECES QUE APARECE
            veces = list(color_labels[i]).count(query_colors)

            # Porcentaje
            porcentaje = (veces / len(color_labels[i])) * 100
            if k_neighbors_percentage != False:
                if porcentaje >= k_neighbors_percentage:
                    porcentajes_list.append([i, porcentaje])
            else:
                porcentajes_list.append([i, porcentaje])

    porcentajes_ordenados = sorted(porcentajes_list, key=lambda x: x[1])  # Ordena segun porcentajes

    print(porcentajes_ordenados)

    for indices in porcentajes_ordenados:
        images_return.append(image_list[indices[0]])

    return images_return

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
    images_return = []
    porcentajes_list = []
    for i, image in enumerate(image_list):
        if query_shape in shape_labels[i]:
            # CONTAMOS VECES QUE APARECE
            veces = list(shape_labels[i]).count(query_shape)

            # Porcentaje
            porcentaje = (veces / len(shape_labels[i])) * 100
            if k_neighbors_percentage != False:
                if porcentaje >= k_neighbors_percentage:
                    porcentajes_list.append([i, porcentaje])
            else:
                porcentajes_list.append([i, porcentaje])

    porcentajes_ordenados = sorted(porcentajes_list, key=lambda x: x[1])  # Ordena segun porcentajes

    print(porcentajes_ordenados)

    for indices in porcentajes_ordenados:
        images_return.append(image_list[indices[0]])

    return images_return

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


    # Retrieval by color
    # Aplicar KMeans
    n_clusters = 5  # Ajusta el número de clusters según sea necesario
    data_means = KMeans(n_clusters=n_clusters)
    data_means.fit(train_imgs)
    centroids = data_means.cluster_centers_

    # Obtener etiquetas de color
    shape_labels = get_colors(centroids)
    query_color = "Red"
    k_neighbors_percentage = False

    # Recuperar imágenes por color
    retrieved_images = retrieval_by_color(train_imgs, shape_labels, query_color, k_neighbors_percentage)

    # Mostrar resultados (opcional)
    for img in retrieved_images:
        plt.imshow(img)
        plt.show()

    # Retrieval by shape
    imgsGray = rgb2gray(train_imgs)
    imgsGray1 = rgb2gray(imgs)
    Knn_test = KNN(imgsGray, train_class_labels)
    k = 5  # Por el momento
    neighbours = Knn_test.get_k_neighbours(imgsGray1, k)

    image_list = imgsGray
    shape_labels = neighbours
    query_shape = "Dresses"
    k_neighbors_percentage = 30

    retrieval_by_shape(image_list, shape_labels, query_shape, k_neighbors_percentage)


    # Retrieval combined
    imgsGray = rgb2gray(train_imgs)
    imgsGray1 = rgb2gray(imgs)
    Knn_test = KNN(imgsGray, train_class_labels)
    k = 5  # Por el momento
    neighbours = Knn_test.get_k_neighbours(imgsGray1, k)
    image_list = imgsGray
    shape_labels = neighbours
    query_shape = "Dresses"

    data_means = KMeans(train_imgs, train_color_labels)
    centroid = data_means.get_centroids()
    shape_labels = km.get_colors(centroid)
    query_color = "Red"
    k_neighbors_percentage = False
    use_percentage = False

    retrieval_combined(image_list, color_labels, shape_labels, query_color, query_shape, use_percentage)



