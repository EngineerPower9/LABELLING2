__authors__ = ["1638317","1634232","1635636"]
__group__ = '1'

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.metrics import accuracy_score

import utl as utl
from utils_data import read_dataset, read_extended_dataset, crop_images, Plot3DCloud, visualize_k_means, visualize_retrieval
from KNN import KNN
from utils import *
from Kmeans import KMeans, distance, get_colors

# Funciones principales de análisis y visualización
def Kmean_statistics(kmeans_class, imatges, Kmax):
    all_wcds, all_iterations, all_times = [], [], []

    options = {'km_init': 'first', 'tolerance': 0.2, 'fitting': 'WCD'}
    fitting_method = options["fitting"]

    for K in range(2, Kmax + 1):
        wcds, iterations, times = [], [], []

        for imatge in imatges:
            km = kmeans_class(imatge, K, options)
            start_time = time.time()
            km.fit()

            if fitting_method == 'WCD':
                km.withinClassDistance()
                metric = km.WCD
            elif fitting_method == 'ICD':
                km.interClassDistance()
                metric = km.ICD
            elif fitting_method == 'FB':
                metric = km.fisherDiscriminant()
            elif fitting_method == 'SF':
                metric = km.silhouetteScore()
            else:
                raise ValueError(f"Error amb el mètode: {fitting_method}")

            num_iter = km.num_iter
            end_time = time.time()
            time_elapsed = end_time - start_time

            wcds.append(metric)
            iterations.append(num_iter)
            times.append(time_elapsed)

        all_wcds.append(np.mean(wcds))
        all_iterations.append(np.mean(iterations))
        all_times.append(np.mean(times))

    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.plot(range(2, Kmax + 1), all_wcds, marker='o')
    plt.title('Fitting by K')
    plt.xlabel('K')
    plt.ylabel('Fitting')

    plt.subplot(132)
    plt.plot(range(2, Kmax + 1), all_iterations, marker='o')
    plt.title('Iterations by K')
    plt.xlabel('K')
    plt.ylabel('Iterations')

    plt.subplot(133)
    plt.plot(range(2, Kmax + 1), all_times, marker='o')
    plt.title('Time to Converge by K')
    plt.xlabel('K')
    plt.ylabel('Time (s)')

    plt.tight_layout()
    plt.show()

def Kmean_best_k(kmeans_class, imatges, Kmax):
    all_fitting, all_iterations, all_times, all_K = [], [], [], []

    options = {'km_init': 'first', 'tolerance': 0.2, 'fitting': 'WCD'}

    for imatge in imatges:
        km = kmeans_class(imatge, 1, options)
        start_time = time.time()
        best_K = km.find_bestK(Kmax)
        end_time = time.time()
        time_elapsed = end_time - start_time

        all_fitting.append(km.WCD)
        all_iterations.append(km.num_iter)
        all_times.append(time_elapsed)
        all_K.append(best_K)

    avg_fitting = np.mean(all_fitting)
    avg_iterations = np.mean(all_iterations)
    avg_times = np.mean(all_times)
    avg_K = np.mean(all_K)

    print(f"El mejor valor promedio de K es: {avg_K:.2f}")
    print(f"Promedio de WCD: {avg_fitting:.2f}")
    print(f"Promedio de Iteraciones: {avg_iterations:.2f}")
    print(f"Promedio de Tiempo (s): {avg_times:.2f}")

    window_length = 5
    polyorder = 2
    smoothed_K = savgol_filter(all_K, window_length, polyorder)
    smoothed_iterations = savgol_filter(all_iterations, window_length, polyorder)
    smoothed_times = savgol_filter(all_times, window_length, polyorder)

    plt.figure(figsize=(16, 4))

    plt.subplot(131)
    plt.plot(range(1, len(smoothed_K) + 1), smoothed_K, marker='o', linestyle='-', color='blue')
    plt.axhline(y=avg_K, color='red', linestyle='--', label=f'Avg K = {avg_K:.2f}')
    plt.title('Best K per Image')
    plt.xlabel('Image Index')
    plt.ylabel('Best K')
    plt.legend()

    plt.subplot(132)
    plt.plot(range(1, len(smoothed_iterations) + 1), smoothed_iterations, marker='o', linestyle='-', color='green')
    plt.axhline(y=avg_iterations, color='red', linestyle='--', label=f'Avg Iterations = {avg_iterations:.2f}')
    plt.title('Iterations per Image')
    plt.xlabel('Image Index')
    plt.ylabel('Iterations')
    plt.legend()

    plt.subplot(133)
    plt.plot(range(1, len(smoothed_times) + 1), smoothed_times, marker='o', linestyle='-', color='purple')
    plt.axhline(y=avg_times, color='red', linestyle='--', label=f'Avg Time = {avg_times:.2f}s')
    plt.title('Time to Converge per Image')
    plt.xlabel('Image Index')
    plt.ylabel('Time (s)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return avg_K

def Kmean_centroids_accuracy(kmeans_class, imatges, true_labels, Kmax):
    all_accuracy = []

    options = {'km_init': 'first', 'tolerance': 0.2}
    resultats = []

    for K in range(2, Kmax + 1):
        accuracy = []

        for imatge in imatges:
            km = kmeans_class(imatge, K, options)
            km.fit()
            accuracy.append(get_colors(km.centroids))

        resultats.append(Get_color_accuracy(accuracy, true_labels))
        all_accuracy.append(resultats[-1])

    plt.figure(figsize=(10, 6))
    plt.plot(range(2, Kmax + 1), all_accuracy, marker='o', linestyle='-', color='b')
    plt.xlabel('Número de Centroides (k)')
    plt.ylabel('Precisión (%)')
    plt.title('Precisión vs Número de Centroides (k)')
    plt.grid(True)
    plt.show()

    return all_accuracy

def Kmean_time_statistics(kmeans_class, imatges, Kmax):
    fitting_methods = ['WCD', 'ICD', 'FB', 'SF']
    method_times = {method: [] for method in fitting_methods}

    for K in range(2, Kmax + 1):
        for method in fitting_methods:
            times = []

            for imatge in imatges:
                options = {'km_init': 'first', 'tolerance': 0.2, 'fitting': method}
                km = kmeans_class(imatge, K, options)
                start_time = time.time()
                km.fit()
                end_time = time.time()
                time_elapsed = end_time - start_time
                times.append(time_elapsed)

            method_times[method].append(np.mean(times))

    plt.figure(figsize=(10, 6))
    for method in fitting_methods:
        plt.plot(range(2, Kmax + 1), method_times[method], marker='o', linestyle='-', label=method)

    plt.xlabel('Número de Centroides (k)')
    plt.ylabel('Tiempo de Convergencia (s)')
    plt.title('Tiempo de Convergencia vs Número de Centroides (k) para Diferentes Métodos')
    plt.legend()
    plt.grid(True)
    plt.show()

def Kmean_time_statistics_2(kmeans_class, imatges, Kmax):
    fitting_methods = ['WCD', 'ICD', 'FB', 'SF']
    init_methods = ['first', 'random', 'diagonal']
    method_times = {init_method: {fit_method: [] for fit_method in fitting_methods} for init_method in init_methods}

    for K in range(2, Kmax + 1):
        for init_method in init_methods:
            for fit_method in fitting_methods:
                times = []

                for imatge in imatges:
                    options = {'km_init': init_method, 'tolerance': 0.2, 'fitting': fit_method}
                    km = kmeans_class(imatge, K, options)
                    start_time = time.time()
                    km.fit()
                    end_time = time.time()
                    time_elapsed = end_time - start_time
                    times.append(time_elapsed)

                method_times[init_method][fit_method].append(np.mean(times))

    plt.figure(figsize=(14, 8))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, init_method in enumerate(init_methods):
        for fit_method in fitting_methods:
            label = f'Init: {init_method}, Fit: {fit_method}'
            plt.plot(range(2, Kmax + 1), method_times[init_method][fit_method], marker='o', linestyle='-', label=label, color=colors[i % len(colors)])

    plt.xlabel('Número de Centroides (k)')
    plt.ylabel('Tiempo de Convergencia (s)')
    plt.title('Tiempo de Convergencia vs Número de Centroides (k) para Diferentes Métodos de Inicialización y Ajuste')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def Kmean_time_statistics_tolerance(kmeans_class, imatges, Kmax):
    fitting_methods = ['WCD', 'ICD', 'FB']
    tolerance_values = [0.05, 0.2, 0.35, 0.6, 0.75]
    method_times = {tol: {fit_method: [] for fit_method in fitting_methods} for tol in tolerance_values}

    for K in range(2, Kmax + 1):
        for tol in tolerance_values:
            for fit_method in fitting_methods:
                times = []

                for imatge in imatges:
                    options = {'km_init': 'first', 'tolerance': tol, 'fitting': fit_method}
                    km = kmeans_class(imatge, K, options)
                    start_time = time.time()
                    km.fit()
                    end_time = time.time()
                    time_elapsed = end_time - start_time
                    times.append(time_elapsed)

                method_times[tol][fit_method].append(np.mean(times))

    plt.figure(figsize=(14, 8))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, tol in enumerate(tolerance_values):
        for fit_method in fitting_methods:
            label = f'Tol: {tol}, Fit: {fit_method}'
            plt.plot(range(2, Kmax + 1), method_times[tol][fit_method], marker='o', linestyle='-', label=label, color=colors[i % len(colors)])

    plt.xlabel('Número de Centroides (k)')
    plt.ylabel('Tiempo de Convergencia (s)')
    plt.title('Tiempo de Convergencia vs Número de Centroides (k) para Diferentes Tolerancias y Métodos de Ajuste')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def Kmean_accuracy_statistics(kmeans_class, imatges, true_labels, Kmax):
    fitting_methods = ['WCD', 'ICD', 'FB', 'SF']
    init_methods = ['first', 'random', 'diagonal']
    accuracy_scores = {init_method: {fit_method: [] for fit_method in fitting_methods} for init_method in init_methods}

    for K in range(2, Kmax + 1):
        for init_method in init_methods:
            for fit_method in fitting_methods:
                accuracies = []

                for imatge in imatges:
                    options = {'km_init': init_method, 'tolerance': 0.2, 'fitting': fit_method}
                    km = kmeans_class(imatge, K, options)
                    km.fit()
                    predicted_labels = get_colors(km.centroids)
                    accuracies.append(Get_color_accuracy(predicted_labels, true_labels))

                accuracy_scores[init_method][fit_method].append(np.mean(accuracies))

    plt.figure(figsize=(14, 8))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, init_method in enumerate(init_methods):
        for fit_method in fitting_methods:
            label = f'Init: {init_method}, Fit: {fit_method}'
            plt.plot(range(2, Kmax + 1), accuracy_scores[init_method][fit_method], marker='o', linestyle='-', label=label, color=colors[i % len(colors)])

    plt.xlabel('Número de Centroides (k)')
    plt.ylabel('Precisión (%)')
    plt.title('Precisión vs Número de Centroides (k) para Diferentes Métodos de Inicialización y Ajuste')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def Kmean_convergence_metrics(kmeans_class, imatges, Kmax):
    fitting_methods = ['WCD', 'ICD', 'FB', 'SF']
    convergence_metrics = {fit_method: [] for fit_method in fitting_methods}

    for K in range(2, Kmax + 1):
        for fit_method in fitting_methods:
            metrics = []

            for imatge in imatges:
                options = {'km_init': 'first', 'tolerance': 0.2, 'fitting': fit_method}
                km = kmeans_class(imatge, K, options)
                km.fit()

                if fit_method == 'WCD':
                    km.withinClassDistance()
                    metric = km.WCD
                elif fit_method == 'ICD':
                    km.interClassDistance()
                    metric = km.ICD
                elif fit_method == 'FB':
                    metric = km.fisherDiscriminant()
                elif fit_method == 'SF':
                    metric = km.silhouetteScore()
                else:
                    raise ValueError(f"Error con el método: {fit_method}")

                metrics.append(metric)

            convergence_metrics[fit_method].append(np.mean(metrics))

    plt.figure(figsize=(14, 8))
    for fit_method in fitting_methods:
        label = f'Método: {fit_method}'
        plt.plot(range(2, Kmax + 1), convergence_metrics[fit_method], marker='o', linestyle='-', label=label)

    plt.xlabel('Número de Centroides (k)')
    plt.ylabel('Valor de la Métrica de Convergencia')
    plt.title('Evolución de las Métricas de Convergencia vs Número de Centroides (k)')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Funciones de cálculo de precisión
def Get_shape_accuracy(predicted_labels, true_labels):
    correct_values = sum(1 for i, label in enumerate(predicted_labels) if label == true_labels[i])
    accuracy = (correct_values / len(predicted_labels)) * 100
    return accuracy

def Get_color_accuracy(predicted_labels, true_labels):
    total_score = 0
    for pred, true in zip(predicted_labels, true_labels):
        intersection = set(pred).intersection(set(true))
        union = set(pred).union(set(true))
        total_score += len(intersection) / len(union) if union else 0
    accuracy = (total_score / len(true_labels)) * 100
    return accuracy

# Funciones de recuperación
def retrieval_by_color(image_list, color_labels, query_colors, use_percentage=False):
    images_return, porcentajes_list = [], []
    for i, image in enumerate(image_list):
        if query_colors in color_labels[i]:
            veces = list(color_labels[i]).count(query_colors)
            porcentaje = (veces / len(color_labels[i])) * 100
            if not use_percentage or porcentaje >= use_percentage:
                porcentajes_list.append([i, porcentaje])
    porcentajes_ordenados = sorted(porcentajes_list, key=lambda x: x[1], reverse=True)
    for indices in porcentajes_ordenados:
        images_return.append(image_list[indices[0]])
    return images_return, porcentajes_ordenados

def retrieval_by_shape(image_list, shape_labels, query_shape, use_percentage=False):
    images_return, porcentajes_list, index = [], [], []
    for i, image in enumerate(image_list):
        if query_shape in shape_labels[i]:
            veces = list(shape_labels[i]).count(query_shape)
            porcentaje = (veces / len(shape_labels[i])) * 100
            if not use_percentage or porcentaje >= use_percentage:
                porcentajes_list.append([i, porcentaje])
    porcentajes_ordenados = sorted(porcentajes_list, key=lambda x: x[1], reverse=True)
    for indices in porcentajes_ordenados:
        images_return.append(image_list[indices[0]])
        index.append(indices[0])
    return images_return, porcentajes_ordenados, index

def retrieval_combined(image_list, color_labels, shape_labels, query_color, query_shape, use_percentage=False):
    shape0, shape1, index = retrieval_by_shape(rgb2gray(image_list), shape_labels, query_shape, use_percentage)
    color0, color1 = retrieval_by_color(image_list, color_labels, query_color, use_percentage)
    index, images_return, porcentajes_ordenados = [], [], []
    for i, shape in enumerate(shape1):
        for j, color in enumerate(color1):
            if color[0] == shape[0]:
                porcentaje = (color1[j][1] * shape1[i][1]) / 100
                index.append(color[0])
                images_return.append(image_list[color[0]])
                porcentajes_ordenados.append([color[0], porcentaje])
    porcentajes_ordenados = sorted(porcentajes_ordenados, key=lambda x: x[1], reverse=True)
    for indices in porcentajes_ordenados:
        images_return.append(image_list[indices[0]])
    return images_return, porcentajes_ordenados

def getBestKforKNN(train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, test_color_labels, classes, imgs, class_labels, color_labels, upper, lower, background, cropped_images):
    imgsGray = rgb2gray(train_imgs)
    imgsGray1 = rgb2gray(test_imgs)
    Knn_test = KNN(imgsGray, train_class_labels)
    calcula = []

    for k in range(1, 20):
        predicted_labels = []
        # Por el momento
        neighbours = Knn_test.get_k_neighbours(imgsGray1, k)
        predicted_labels = Knn_test.get_class()
        accuracy = Get_shape_accuracy(predicted_labels, test_class_labels)
        calcula.append(accuracy)

    print(calcula)
    min_value = max(calcula)
    best_k = calcula.index(min_value)
    print("BEST: ", best_k + 1)
    print("VALUE: ", calcula[best_k])
    return min_value

# Funciones de test
def test_retrieval_by_color(train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, test_color_labels, classes, imgs, class_labels, color_labels, upper, lower, background, cropped_images):
    labels_function, options = [], {'km_init': 'diagonal', 'tolerance': 0.0, 'fitting': 'WCD'}
    for analize in cropped_images:
        km = KMeans(analize, 1, options)
        KMeans.find_bestK(km, len(colors))
        labels = get_colors(km.centroids)
        labels_function.append(labels)
    query_color = "Red"
    result_imgs, result_info = retrieval_by_color(imgs, labels_function, query_color, False)
    visualize_retrieval(result_imgs, 16, result_info, None, 'Resultados por Color')
    predicted_labels = labels_function[:len(test_color_labels)]
    color_accuracy = Get_color_accuracy(predicted_labels, color_labels)
    print(f'Color Accuracy: {color_accuracy:.2f}%')

def test_retrieval_by_shape(train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, test_color_labels, classes, imgs, class_labels, color_labels, upper, lower, background, cropped_images):
    imgsGray, imgsGray1 = rgb2gray(train_imgs), rgb2gray(test_imgs)
    Knn_test = KNN(imgsGray, train_class_labels)
    k = 15
    neighbours = Knn_test.get_k_neighbours(imgsGray1, k)
    shape_labels = neighbours
    query_shape = "Shorts"
    result_imgs, result_info, index = retrieval_by_shape(imgsGray1, neighbours, query_shape, False)
    result_imgs_show = [test_imgs[i] for i in index]
    visualize_retrieval(result_imgs_show, 16, result_info, None, 'Resultados por Forma')
    predicted_shape_labels = Knn_test.get_class()
    shape_accuracy = Get_shape_accuracy(predicted_shape_labels, test_class_labels)

def test_retrieval_combined(train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, test_color_labels, classes, imgs, class_labels, color_labels, upper, lower, background, cropped_images):
    labels_function = []
    for analize in imgs:
        options = {}
        km = KMeans(analize)
        KMeans.find_bestK(km, len(colors))
        labels = get_colors(km.centroids)
        labels_function.append(labels)
    imgsGray, imgsGray1 = rgb2gray(train_imgs), rgb2gray(test_imgs)
    Knn_test = KNN(imgsGray, train_class_labels)
    k = 30
    neighbours = Knn_test.get_k_neighbours(imgsGray1, k)
    shape_labels = neighbours
    query_shape, query_color = "Dresses", "Red"
    result_imgs, result_info = retrieval_combined(test_imgs, test_color_labels, shape_labels, query_color, query_shape, False)
    visualize_retrieval(result_imgs, 10, result_info, None, 'Combinados')

def test_Get_shape_accuracy(train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, test_color_labels, classes, imgs, class_labels, color_labels, upper, lower, background, cropped_images):
    imgsGray, imgsGray1 = rgb2gray(train_imgs), rgb2gray(test_imgs)
    Knn_test = KNN(imgsGray, train_class_labels)
    k = 15
    neighbours = Knn_test.get_k_neighbours(imgsGray1, k)
    predicted_labels = Knn_test.get_class()
    accuracy = Get_shape_accuracy(predicted_labels, test_class_labels)
    print("SHAPE ACCURACY: ", accuracy)
    return accuracy

def test_Get_color_accuracy(train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, test_color_labels, classes, imgs, class_labels, color_labels, upper, lower, background, cropped_images):
    labels_function = []
    for analize in imgs:
        options = {}
        km = KMeans(analize)
        KMeans.find_bestK(km, len(colors))
        labels = get_colors(km.centroids)
        labels_function.append(labels)
    accuracy = Get_color_accuracy(labels_function, color_labels)
    print("Get color accuracy: ", accuracy)
    return accuracy

# Funciones de carga de datos
def train_load():
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')
    classes = list(set(list(train_class_labels) + list(test_class_labels)))
    return train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, test_color_labels, classes

def truth_load():
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)
    return imgs, class_labels, color_labels, upper, lower, background, cropped_images

if __name__ == '__main__':
    # Inicialización de variables
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, test_color_labels, classes = train_load()
    imgs, class_labels, color_labels, upper, lower, background, cropped_images = truth_load()

    # TEST RETRIEVAL BY COLOR
    # test_retrieval_by_color(train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, test_color_labels, classes, imgs, class_labels, color_labels, upper, lower, background, cropped_images)
    # Kmean_statistics(KMeans, imgs, len(colors))

    # TEST RETRIEVAL BY SHAPE
    # test_retrieval_by_shape(train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, test_color_labels, classes, imgs, class_labels, color_labels, upper, lower, background, cropped_images)

    # TEST RETRIEVAL COMBINED
    # test_test_retrieval_combined(train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, test_color_labels, classes, imgs, class_labels, color_labels, upper, lower, background, cropped_images)

    # TEST SHAPE_ACCURACY
    # test_Get_shape_accuracy(train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, test_color_labels, classes, imgs, class_labels, color_labels, upper, lower, background, cropped_images)

    # Test COLOR ACCURACY
    # test_Get_color_accuracy(train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, test_color_labels, classes, imgs, class_labels, color_labels, upper, lower, background, cropped_images)

    # TEST BESTKFOR KNN
    #getBestKforKNN(train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, test_color_labels, classes, imgs, class_labels, color_labels, upper, lower, background, cropped_images)

    # Llamadas a las funciones de prueba
    # Kmean_statistics(KMeans, imgs, len(colors))
    # Kmean_best_k(KMeans, imgs, len(colors))
    # Kmean_centroids_accuracy(KMeans, imgs, color_labels, len(colors))
    # Kmean_time_statistics_2(KMeans, imgs, len(colors))
    # Kmean_time_statistics_tolerance(KMeans, imgs, len(colors))
    # Kmean_accuracy_statistics(KMeans, imgs, color_labels, len(colors))
    #Kmean_convergence_metrics(KMeans, imgs, len(colors))
    # test_retrieval_by_color(train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, test_color_labels, classes, imgs, class_labels, color_labels, upper, lower, background, cropped_images)
    # test_retrieval_by_shape(train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, test_color_labels, classes, imgs, class_labels, color_labels, upper, lower, background, cropped_images)
    # test_retrieval_combined(train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, test_color_labels, classes, imgs, class_labels, color_labels, upper, lower, background, cropped_images)
    # test_Get_shape_accuracy(train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, test_color_labels, classes, imgs, class_labels, color_labels, upper, lower, background, cropped_images)
    # test_Get_color_accuracy(train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, test_color_labels, classes, imgs, class_labels, color_labels, upper, lower, background, cropped_images)
