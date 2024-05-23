__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

import utl as utl
from utils_data import read_dataset, read_extended_dataset, crop_images, Plot3DCloud, visualize_k_means, \
    visualize_retrieval
import time
import matplotlib.pyplot as plt
import numpy as np
from KNN import __authors__, __group__, KNN
from utils import *
from Kmeans import __authors__, __group__, KMeans, distance, get_colors


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
                    #print(porcentajes_list)
            else:
                porcentajes_list.append([i, porcentaje])

    porcentajes_ordenados = sorted(porcentajes_list, key=lambda x: x[1], reverse=True)  # Ordena segun porcentajes

    #print(porcentajes_ordenados)

    for indices in porcentajes_ordenados:
        images_return.append(image_list[indices[0]])

    return images_return, porcentajes_ordenados

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
    index = []
    for i, image in enumerate(image_list):
        if query_shape in shape_labels[i]:
            #CONTAMOS VECES QUE APARECE
            veces = list(shape_labels[i]).count(query_shape)

            #Porcentaje
            porcentaje = (veces/len(shape_labels[i]))*100
            if k_neighbors_percentage != False:
                if porcentaje >= k_neighbors_percentage:
                    porcentajes_list.append([i, porcentaje])
            else:
                porcentajes_list.append([i, porcentaje])


    porcentajes_ordenados = sorted(porcentajes_list, key=lambda x: x[1], reverse=True) #Ordena segun porcentajes


    #print(porcentajes_ordenados)

    for indices in porcentajes_ordenados:
        images_return.append(image_list[indices[0]])
        index.append(indices[0])

    return images_return, porcentajes_ordenados, index

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
    shape0, shape1 = retrieval_by_shape(image_list, shape_labels, query_shape, use_percentage)
    color0, color1 = retrieval_by_color(image_list, color_labels, query_color, use_percentage)

    #Color1 y shape1 contiene los porcentajes con los indices de las imagenes.
    #color0 y shape0 contiene las imagenes correspondientes.
    #print(shape1, color1)
    index = []
    images_return = []
    porcentajes_ordenados = []

    for i, shape in enumerate(shape1):
        for j, color in enumerate(color1): #Si la imatge està en els dos casos la retornarà
            if color[0] == shape[0]:
                index.append(color[0])
                images_return.append(image_list[color[0]])
                porcentajes_ordenados.append(color0[j]*shape0[i])

    return images_return, porcentajes_ordenados

def calculate_accuracy(predictions, ground_truth):
    correct = np.sum(predictions == ground_truth)
    return correct / len(ground_truth) * 100

if __name__ == '__main__':
    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
    test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')
    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

#########################test_retrieval_by_color()#########################

# Crea una instancia de KMeans con los parámetros adecuados
    labels_function = []
    for analize in imgs:
        options = {}
        colors_list = []
        km = KMeans(analize)
        KMeans.find_bestK(km, len(colors))
        labels = get_colors(km.centroids)
        labels_function.append(labels)

    # Recuperación por color
    query_color = "Blue"
    k_neighbors_percentage = False
    result_imgs, result_info = retrieval_by_color(imgs, labels_function, query_color, k_neighbors_percentage)
    print(result_imgs)
    # Visualización
    visualize_retrieval(result_imgs, 10, result_info, None, 'Resultados por Color')

    # Calcular precisión de color
    predicted_labels = labels_function[:len(test_color_labels)]  # Asegúrate de alinear las longitudes
    color_accuracy = Get_color_accuracy(predicted_labels, color_labels)
    print(f'Color Accuracy: {color_accuracy:.2f}%')

###########################################################################

#########################test_retrieval_by_shape()#########################

    imgsGray = rgb2gray(train_imgs)
    imgsGray1 = rgb2gray(imgs)

    Knn_test = KNN(imgsGray, train_class_labels)
    k = 15 #Por el momento
    neighbours = Knn_test.get_k_neighbours(imgsGray1, k)

    #image_list = imgsGray1
    shape_labels = neighbours

    # Recuperación por forma
    query_shape = "Dresses"
    k_neighbors_percentage = False
    result_imgs, result_info,index = retrieval_by_shape(imgsGray1, neighbours, query_shape)
    result_imgs_show = []
    for i in index:
        result_imgs_show.append(imgs[i])
    # Visualización

    visualize_retrieval(result_imgs_show, 10, result_info, None, 'Resultados por Forma')

    # Calcular precisión de forma
    predicted_shape_labels = neighbours[:len(test_class_labels)]  # Asegurarse de alinear las longitudes
    shape_accuracy = Get_shape_accuracy(predicted_shape_labels, test_class_labels)
    #print(f'Shape Accuracy: {shape_accuracy:.2f}%')

###########################################################################

#########################test_retrieval_combined()#########################
    labels_function = []
    for analize in imgs:
        options = {}
        colors_list = []
        km = KMeans(analize)
        KMeans.find_bestK(km, len(colors))
        labels = get_colors(km.centroids)
        labels_function.append(labels)

    imgsGray = rgb2gray(train_imgs)
    imgsGray1 = rgb2gray(imgs)
    Knn_test = KNN(imgsGray1, train_class_labels)
    k = 5  # Por el momento
    neighbours = Knn_test.get_k_neighbours(imgsGray1, k)
    image_list = imgs
    shape_labels = neighbours
    query_shape = "Dresses"
    query_color = "Red"
    use_percentage = False

    result_imgs, result_info = retrieval_combined(image_list, color_labels, shape_labels, query_color, query_shape, use_percentage)
    visualize_retrieval(result_imgs, 10, result_info, None, 'Combinados')

###########################################################################