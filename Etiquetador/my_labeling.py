__authors__ = '1497706, 1430797, 1467542'
__group__ = 'DM.18'

import utils as utl
import numpy as np
import Kmeans as km
import KNN as knn
from utils_data import read_dataset, visualize_k_means, visualize_retrieval, Plot3DCloud, read_one_img


# Test Find_BestK modificando el umbral de calculo de la K de KMeans
def diccionarioRopa(classes, datos):
    # bucle para contar y clasificar los errores y lista de prints para imprimir el diccionario
    for x in classes:
        shirts = 0
        dresses = 0
        shorts = 0
        sandals = 0
        flip_flops = 0
        handbags = 0
        jeans = 0
        socks = 0
        heels = 0
        for i in datos[x]:
            if i == "Shirts":
                shirts += 1
            if i == "Dresses":
                dresses += 1
            if i == "Sandals":
                sandals += 1
            if i == "Flip Flops":
                flip_flops += 1
            if i == "Handbags":
                handbags += 1
            if i == "Jeans":
                jeans += 1
            if i == "Socks":
                socks += 1
            if i == "Shorts":
                shorts += 1
            if i == "Heels":
                heels += 1
        print(x)
        print("shirts:")
        print(shirts)
        print("dresses:")
        print(dresses)
        print("sandals:")
        print(sandals)
        print("flip_flops:")
        print(flip_flops)
        print("handbags:")
        print(handbags)
        print("jeans:")
        print(jeans)
        print("socks:")
        print(socks)
        print("shorts:")
        print(shorts)
        print("heels:")
        print(heels)


def change_tolarance(test_imgs, option):
    print("---------Calculo de la K ideal---------")
    for x in test_imgs:
        k_posible = []
        print("-------")
        for i in range(1, 10):
            option['tolerance'] = i / 10
            K_aux = km.KMeans(x, 1, option)
            km.KMeans.select_findK_method(K_aux, 9)
            k_posible.append(K_aux.K)
            print("Tolerancia:" + str(option['tolerance']) + " Mejor K obtenida:" + str(K_aux.K))
            if k_posible.count(K_aux.K) > 2:
                K_millor = km.KMeans(x, K_aux.K, option)
                km.KMeans.fit(K_millor)
                break
            if i == 9:
                npK_posible = np.array(k_posible)
                bestK = np.amin(npK_posible)
                K_millor = km.KMeans(x, bestK, option)
                km.KMeans.fit(K_millor)


def retrieval_by_shape(lista_formas, query):
    index = []
    i = 0
    for image in lista_formas:
        if query == image:
            index.append(i)
        i += 1
    return index


def get_shape_accuracy(lista_formas, query, test_shape_labels):
    indices_knn = retrieval_by_shape(lista_formas, query)
    indices_ground_truth = retrieval_by_shape(test_shape_labels, query)
    aciertos = 0
    errores = 0
    for image in indices_knn:
        if image in indices_ground_truth:
            aciertos = aciertos + 1
        else:
            errores = errores + 1
    Total = len(indices_ground_truth)
    misses = Total - aciertos
    percentateAciertos = aciertos / Total
    percentateMisses = misses / Total
    pocentajeErrores = errores / Total
    print(aciertos, misses, errores)
    print("Aciertos:" + str(percentateAciertos) + " Misses:" + str(percentateMisses) + " Errores:" + str(
        pocentajeErrores))


def retrieval_by_color(lista_colores, query):
    index = []
    imagenes = []
    i = 0
    for image in lista_colores:
        if query in image:
            index.append(i)
        i += 1
    return index


def test_color_accuracy(lista_colores, query, test_color_labels):
    indices_kmeans = retrieval_by_color(lista_colores, query)
    indices_ground_truth = retrieval_by_color(test_color_labels, query)
    aciertos = 0
    errores = 0
    for image in indices_kmeans:
        if image in indices_ground_truth:
            aciertos = aciertos + 1
        else:
            errores = errores + 1
    Total = len(indices_ground_truth)
    misses = Total - aciertos
    percentateAciertos = aciertos / Total
    percentateMisses = misses / Total
    pocentajeErrores = errores / Total
    print(aciertos, misses, errores)
    print("Aciertos:" + str(percentateAciertos) + " Misses:" + str(percentateMisses) + " Errores:" + str(
        pocentajeErrores))


if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
    test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    # List with all the existant classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    option = {}
    datos = {}
    option['km_init'] = 'colors'
    option['fitting'] = 'fisher'
    option['tolerance'] = 0.2

    predict_colores_KMeans = []
    list_total = []

    # Test Shape - Analisis cualitativo y cuantitativo.
    knn_list_train = knn.KNN(train_imgs, train_class_labels)
    lista_formas = knn_list_train.predict(test_imgs, 3)
    for roba in classes:
        # print("---------" + str(roba) + "--------")
        get_shape_accuracy(lista_formas, roba, test_class_labels)

        # codigo para gardar en un diccionario el resultado de la bçusqueda
        index_roba = retrieval_by_shape(lista_formas, roba)
        resultats = []
        for index in index_roba:
            resultats.append(test_class_labels[index])
        datos[roba] = resultats
    diccionarioRopa(classes, datos)
    # Analizar K en funcion a la tolerancia
    change_tolarance(test_imgs, option)

    print("---------Calculo de Colores---------")
    for x in test_imgs:
        K_aux = km.KMeans(x, 1, option)
        km.KMeans.find_bestK(K_aux, 4)  # Se ha modificado el valor de K_Max para obtener diferentes resultados
        km.KMeans.fit(K_aux)
        predict_colores_KMeans.append(km.get_colors(K_aux.centroids))

    for color in utl.colors:
        index = []
        imagenes = []
        print("---------" + str(color) + "--------")
        i = 0
        index = retrieval_by_color(predict_colores_KMeans, color)
        for c in index:
            imagenes.append((test_imgs[c]))
        test_color_accuracy(predict_colores_KMeans, color, test_color_labels)
