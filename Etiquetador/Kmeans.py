__authors__ = ["1638317","1634232","1635636"]
__group__ = 'noneyet'

import numpy as np
import utils
from sklearn.metrics import silhouette_score

class KMeans:
    # Incialització de la classe kmeans
    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        # Paràmetres de la funció kmeans
        self.num_iter = 0  # Nombre d'iteracions que farem per buscar la millor distribució de classes
        self.K = K  # Nombre de classes a buscar
        self._init_X(X)  # Matriu de RGB
        self._init_options(options)  # DICT options que decideixen com modifica la forma de buscar classes

    # Funció que inicialitza la matriu de punts
    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """
        # Primer mirem si está en forma de numpy array
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        # Convertim els valors del array en float64, és a dir, floats normals (si son d'algun altre tipus)
        X = X.astype(np.float64)

        # Si la dimensió del array es major a 2 (F*C*3), la transformen a una de dimensió 2 (N*3) on N=F*C
        if X.ndim > 2:
            # Funció que automàticament ajusta la matriu per a que capigue la F*C
            X = X.reshape(-1, X.shape[-1]) # Lleugera millora

        # Afegim la nova matriu a la classe
        self.X = X

    # Inicialitza els paràmetres de KMeans
    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        # Opcions per modificar el kmeans que es guarden en format diccionari
        if options is None:
            options = {}
        if 'km_init' not in options:
            # Modifica la lògica del algorisme usada per fer el kmeans y buscar veins al inici
            options['km_init'] = 'first'
        if 'verbose' not in options:
            # Flag usada per permetre missatges de control sobre els resultats
            options['verbose'] = False
        if 'tolerance' not in options:
            # Adjusta la tolerancia: Para l'algorisme quan el la diferencia entre el old centroid i el nou es de 0
            options['tolerance'] = 0.2
        if 'max_iter' not in options:
            # Quantitat màxima d'iteracions permeses pel kmeans per trobar les classes / solució
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            # Usa una fòrmula per mirar quan un algorisme es pot considerar que arriba al nombre de k ideal/òptim
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

    # Inicialitza les classes centròids i centròids antics
    def _init_centroids(self):
        """
        Initialization of centroids
        """
        # Si la opció first apareix, anirem mirant de la taula, cada combinació RGB en l'ordre que apareixen
        if self.options['km_init'].lower() == 'first':
            # Creem un set de files úniques
            unics = set()
            centroids = []
            # Anem fila per fila mirant si està inclosa cada combinació de RGB
            for row in self.X:
                fila = tuple(row)
                # Si trobem punts amb combinació RGB no incloses els incloem
                if fila not in unics:
                    unics.add(fila)
                    centroids.append(row)
                    # Comprobem que no ens passem de classes
                    if len(centroids) == self.K:
                        break
            # Si no obtenim suficients punts per a totes les classes que volem, provoquem un error
            if len(centroids) < self.K:
                raise ValueError("NO tenim suficients punts!!!")

            # Si tot funciona correctament i tenim suficients punts, afegim els centroides
            self.centroids = np.array(centroids)

        # Opció on agafem de la nostra matriu de punts, punts aleatoris per iniciar
        elif self.options['km_init'].lower() == 'random':
            self.centroids = self.X[np.random.choice(len(self.X), self.K, replace=False)]

        # Format custom
        elif self.options['km_init'].lower() == 'diagonal':
            # Obtenim els valors mínims i màxims per cada dimensió de les dades
            min_vals = np.min(self.X, axis=0)
            max_vals = np.max(self.X, axis=0)

            # Inicialitzem una llista per guardar els centroides
            centroids = []

            # Calculem la separació entre centroides al llarg de la diagonal
            step = (max_vals - min_vals) / (self.K - 1)

            # Generem els centroides al llarg de la diagonal
            for i in range(self.K):
                centroid = min_vals + i * step
                centroids.append(centroid)

            self.centroids = np.array(centroids)

    # Assigna el centròid més proper
    def get_labels(self):
        """
        Calculates the closest centroid of all points in X and assigns each point to the closest centroid
        """
        # Calculem la distància de tots els punts de la matriu i els centroides escollits
        distances = distance(self.X, self.centroids)
        # Després afegim el que tingui menys distància
        self.labels = np.argmin(distances, axis=1)

    # Reassigna centroids
    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        # Copiem els antics centroides a una variable
        self.old_centroids = np.copy(self.centroids)

        for i in range(len(self.centroids)):
            assigned_points = self.X[self.labels == i]
            # Si tenim punts assignats, agafem la distància mitja
            if assigned_points.size > 0:
                self.centroids[i] = np.mean(assigned_points, axis=0)
            # Si no hi ha punts assignats, simplement deixem els antics centroides
            else:
                self.centroids[i] = self.old_centroids[i]

    # Verifica si l'algorisme ha arribat al final
    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        # Funció que comprova si els antics centroides son iguals o no als nous mirant la seva diferència
        return np.allclose(self.centroids, self.old_centroids, atol=self.options["tolerance"], rtol=0)

    # Executa l'algorisme KMeans
    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number of iterations is smaller
        than the maximum number of iterations.
        """
        # Iniciem els primers centroides y paràmetres per fer funcionar l'algorisme
        self._init_centroids()
        has_converged = False
        iter_count = 0

        # Mentre que no convergeixi ni es passi del màxim d'iteracions fem que funcioni
        while not has_converged and iter_count < self.options['max_iter']:
            # Obtenim les etiquetes per a la imatge
            self.get_labels()
            # Obtenim la posició del centroides que defineixen les classes
            self.get_centroids()
            # Mirem si la distància de classe es
            has_converged = self.converges()
            # Si no l'obtenim, passem a la següent iteració
            iter_count += 1

        self.num_iter += iter_count

    # Calcula la intra-class o WCD
    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """
        # Revisa segons la fòrmula donada si la diferència entre distpancia interclass arriba al % suficient
        distances = distance(self.X, self.centroids)
        min_distances = np.min(distances, axis=1)
        self.WCD = np.sum(np.square(min_distances)) / len(self.X)

    # Calcula la inter-class o ICD
    def interClassDistance(self):
        interclass = 0.0
        for i in range(len(self.centroids)):
            assigned_points = self.X[self.labels == i]
            if assigned_points.size > 0:
                distances = np.linalg.norm(assigned_points - self.centroids[i], axis=1)
                interclass += np.sum(distances)
        self.ICD = interclass / self.K

    # Calcula el discriminant de fisher
    def fisherDiscriminant(self):
        self.withinClassDistance()
        self.interClassDistance()
        self.FD = self.ICD / self.WCD
        return self.FD

    def silhouetteScore(self):
        """
        Calculates the silhouette score of the current clustering
        """
        self.silhouette = silhouette_score(self.X, self.labels)
        return self.silhouette

    # Troba el valor K òptim
    def find_bestK(self, max_K):
        list_metrics = []
        tolerance = self.options['tolerance']
        fitting_method = "WCD"

        for i in range(2, max_K + 1):
            self.K = i
            self.fit()

            # Designem la condició de fitting més adequada a cada cas
            if fitting_method == 'WCD':
                self.withinClassDistance()
                metric = self.WCD
            elif fitting_method == 'ICD':
                self.interClassDistance()
                metric = self.ICD
            elif fitting_method == 'FB':
                metric = self.fisherDiscriminant()
            elif fitting_method == 'SF':
                metric = self.silhouetteScore()
            else:
                raise ValueError(f"Error amb el mètode: {fitting_method}")

            list_metrics.append(metric)

            if i > 2:
                reduction = (list_metrics[-2] - list_metrics[-1]) / list_metrics[-2]
                if 0 < reduction < tolerance:
                    self.K = i - 1
                    break

        return self.K

# Calcula la distància entre cada píxel i cada centròid
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
    # Preparem els operands per usar pitàgores y ajustem les dimensions per tenir bé els càlculs
    X2 = np.sum(np.square(X), axis=1, keepdims=True)
    C2 = np.sum(np.square(C), axis=1)
    # Apliquem la fòrmula de pitàgores per veure la disància però usant matrius
    distancies = np.sqrt(X2 - 2 * np.dot(X, C.T) + C2)

    # Retornem una matriu de les distàncies de cada centroide respecte cada punt de la nostra matriu
    return distancies

# Per a cada centròid retorna el color de l'etiqueta
def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """
    # Retorna els colors que predominen per identificar i retorna les seves etiquetes en funció d'aquests
    X = np.argmax(utils.get_color_prob(centroids), axis=1)
    return [utils.colors[i] for i in X]

