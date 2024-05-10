__authors__ = ["1638317","1634232","1635636"]
__group__ = 'noneyet'

import numpy as np
import utils


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options


    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        X = X.astype(np.float64)

        if X.ndim > 2:
            X = X.reshape(np.prod(X.shape[:-1]), X.shape[-1])

        self.X = X


    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

    def _init_centroids(self):
        """
        Initialization of centroids
        """

        if self.options['km_init'].lower() == 'first':
            unics = set()
            centroids = []
            for row in self.X:
                fila = tuple(row)
                if fila not in unics:
                    unics.add(fila)
                    centroids.append(row)
                    if len(centroids) == self.K:
                        break
            if len(centroids) < self.K:
                raise ValueError("NO tenim suficients punts!!!")
            self.centroids = np.array(centroids)

        elif self.options['km_init'].lower() == 'random':
            self.centroids = self.X[np.random.choice(len(self.X), self.K, replace=False)]

        elif self.options['km_init'].lower() == 'custom':
            pass


    def get_labels(self):
        """
        Calculates the closest centroid of all points in X and assigns each point to the closest centroid
        """
        distances = distance(self.X, self.centroids)
        self.labels = np.argmin(distances, axis=1)


    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        self.old_centroids = np.copy(self.centroids)
        for i in range(len(self.centroids)):
            assigned_points = self.X[self.labels == i]
            if assigned_points.size > 0:
                self.centroids[i] = np.mean(assigned_points, axis=0)
            else:
                self.centroids[i] = self.old_centroids[i]


    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        return np.allclose(self.centroids, self.old_centroids, atol=self.options["tolerance"], rtol=0)


    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number of iterations is smaller
        than the maximum number of iterations.
        """
        self._init_centroids()
        has_converged = False
        iter_count = 0

        while not has_converged and iter_count < self.options['max_iter']:
            self.get_labels()
            self.get_centroids()
            has_converged = self.converges()
            iter_count += 1


    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """
        distances = distance(self.X, self.centroids)
        min_distances = np.min(distances, axis=1)
        self.WCD = np.sum(np.square(min_distances)) / len(self.X)


    def find_bestK(self, max_K):
        """
         sets the best k analysing the results up to 'max_K' clusters
        """
        listWCD = []
        for i in range(2, max_K + 1):
            self.K = i
            self.fit()
            self.withinClassDistance()
            listWCD.append(self.WCD)

            if i > 2:
                reduction = (listWCD[-2] - listWCD[-1]) / listWCD[-2]
                if 0 < reduction < 0.2:
                    self.K = i - 1
                    break
        else:
            self.K = max_K


def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """
    X2 = np.sum(np.square(X), axis=1, keepdims=True)
    C2 = np.sum(np.square(C), axis=1)
    distancies = np.sqrt(X2 - 2 * np.dot(X, C.T) + C2)  # Applying the distance formula

    return distancies


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """
    X = np.argmax(utils.get_color_prob(centroids), axis=1)
    return [utils.colors[i] for i in X]


