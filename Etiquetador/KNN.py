__authors__ = ["1638317","1634232","1635636"]
__group__ = '1'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images --> P imatges amb MxN dimensions
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        #self.train_data = np.random.randint(8, size=[10, 4800])#Inicialització de return

        imgs = []
        for img in train_data:
            valueimg = []
            for filas in img:
                for values in filas:
                    valueimg.append((values))
            imgs.append(valueimg)

        self.train_data = np.array(imgs)

    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        imgs = []
        for img in test_data:
            valueimg = []
            for filas in img:
                for values in filas:
                    valueimg.append((values))
            imgs.append(valueimg)

        #CÀLCUL DE DISTÀNCIES
        distances = cdist(imgs, self.train_data)

        #K CLASSES
        #Evaluem per imatges
        returns = []
        i = 1
        for img in distances:
            i = i+1
            aux = img
            minK = []

            for s in range(k):
                minVal = min(aux)
                # ELIMINAR VALOR DE AUX
                indexElim = np.where(aux == minVal)[0]
                indexReal = np.where(img == minVal)[0]
                minK.append(self.labels[indexReal][0])
                aux = np.delete(aux, indexElim)
            returns.append(minK)

        self.neighbors = np.array(returns)
        #self.neighbors = np.random.randint(k, size=[test_data.shape[0], k])
        return self.neighbors

    def get_class(self):
        """
        Get the class by maximum voting
        :return: 1 array of Nx1 elements. For each of the rows in self.neighbors gets the most voted value
                (i.e. the class at which that row belongs)
        """

        values_res = []
        for img in self.neighbors:
            #Primer comprovem si es repeteix algun valor
            valors_No_repetits, vegades = np.unique(img, return_counts=True)

            if len(valors_No_repetits) == len(img):
                #No s'han repetit valors i ens quedem amb el primer
                values_res.append(img[0])

            else:
                #Trobar el valor màxim de vegades al array vegades.
                max_val = np.max(vegades)
                valors = [] #Per si hi ha més d'un
                for i, value in enumerate(valors_No_repetits):
                    if vegades[i] == max_val:
                        valors.append(value)

                #Busquem l'ordre en la llista original en cas de ser més d'un
                if len(valors) > 1:
                    for auxiliar in img:
                        if auxiliar in valors:
                            values_res.append(auxiliar)
                            break
                else:
                    values_res.append(valors[0])
        print(np.array(values_res))
        return np.array(values_res)




        #return np.random.randint(10, size=self.neighbors.size), np.random.random(self.neighbors.size)

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class a Nx1 vector with the predicted shape for each test image
        """
        self.get_k_neighbours(test_data, k)
        return self.get_class()