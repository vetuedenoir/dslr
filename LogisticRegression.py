import numpy as np


class LogisticRegression():
    """
    logistic regression to help classify things.
    """

    def __init__(self, theta, alpha=0.001, max_iter=1000, algo='gradient_descent', batch_size=128, optimization=""):
        """
        Initialize a Linear Regression model.
        Parameters:
        theta : numpy.ndarray
            Model parameters (weights), of shape (n_features + 1, 1).
            The first element corresponds to the bias term.

        alpha : float, optional (default=0.001)
            Learning rate used to update the model parameters during training.

        max_iter : int, optional (default=1000)
            Maximum number of iterations for the optimization algorithm.

        algo : str, optional (default='gradient_descent')
            Optimization algorithm used to minimize the cost function.
            Supported values are:
            - 'gradient_descent' or 'GD'
            - 'stochastic_gradient_descent' or 'SGD'
            - 'mini-batch_gradient_descent' or 'mbGD'

        batch_size : int, optional (default=128)
            Size of the mini-batches used for mini-batch gradient descent.
            Ignored for full gradient descent.

        optimization : str, optional
            Optional optimization strategy (e.g. 'momentum', 'adam', etc.).
            If empty, standard gradient descent is used.

        Raises:
            If theta is not a numpy.ndarray or does not have shape (n + 1, 1).
        """

        if not isinstance(theta, np.ndarray):
            raise Exception("theta must be a numpy.ndarray")
        if len(theta.shape) != 2 or theta.shape[1] != 1:
            raise Exception("theta must be a column vector of shape (n + 1, 1)")

        match algo:
            case "gradient_descent" | "GD":
                algo = self.gradient_descent
            case "stochastic_gradient_descent" | "stochastic_GD" | "SGD":
                algo = self.stochastic_gradient_descent
            case "mini-batch_gradient_descent" | "mini-batch_GD" | "mbGD":
                algo = self.miniBatch_gradient_descent
            case _:
                print(f"Unknown algorithm: {algo}, using gradient descent.")
                algo = self.gradient_descent

        self.theta = theta
        self.alpha = alpha
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.algo = algo
        self.optimization = optimization
        self.historique = []
        # l'historique de la loss pour voir la vitesse d'aprentissage
        # On pourra l'afficher sur un graphique


    def sigmoid_(self, x):
        """
        Compute the sigmoid of a vector.
        Args:
            x: has to be a numpy.ndarray of shape (m, 1).
        Returns:
            The sigmoid value as a numpy.ndarray of shape (m, 1).
        Raises:
            This function should not raise any Exception.
        """
        return 1 / (1 + 1 / np.exp(x))


    def log_predict_(self, x):
        """
        Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
        Args:
            x: has to be an numpy.ndarray, a vector of dimension m * n.
        Returns:
            y_hat as a numpy.ndarray, a vector of dimension m * 1.
        Raises:
            This function raise Exeptions on any error.
        """
        if not isinstance(x, np.ndarray):
            raise Exception("x as to be a numpy.ndarray")
        if len(x.shape) != 2:
            raise Exception("x has to be a vector of shape m * 1.")

        m, n = x.shape

        if self.theta.shape[0] != n + 1:
            raise Exception("x has to be a vector of shape m * 1.")
        
        Xprime = np.c_[np.ones((m, 1)), x]

        # return 1 / (1 + 1 / np.exp(Xprime @ self.theta))
        return self.sigmoid_(Xprime @ self.theta)


    def log_loss_(self, y, y_hat, eps=1e-15):
        """
        Compute the logistic loss value.
        Args:
            y: has to be an numpy.ndarray, a vector of shape m * 1.
            y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
            eps: epsilon (default=1e-15)
        Returns:
            The logistic loss value as a float.
        Raises:
            This function raise Exeptions on any error.
        """

        if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray) or not isinstance(eps, float):
            raise Exception("y and y_hat must be a numpy.ndarray, and eps a float")
        if len(y.shape) != 2 or len(y_hat.shape) != 2:
            raise Exception("y and y_hat must be a vector of shape m * 1.")
        if y.shape[1] != 1 or y_hat.shape[1] != 1 or y.shape[0] != y_hat.shape[0]:
            raise Exception("y and y_hat must be a vector of shape m * 1.")

        m, n = y.shape
        y_hat = np.clip(y_hat, eps, 1 - eps)

        return -np.mean((y * np.log(y_hat) + (np.ones(m) - y) * np.log(np.ones(m) - y_hat)))


    def log_gradient(self, x, y):
        """
        Computes a gradient vector from three non-empty numpy.ndarray.
        Args:
            x: has to be an numpy.ndarray, a matrix of shape m * n.
            y: has to be an numpy.ndarray, a vector of shape m * 1.
        Returns:
            The gradient as a numpy.ndarray, a vector of shape n * 1.
        Raises:
            This function raise Exeption on any error.
        """
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise Exception("x and y must be a numpy.ndarray, and eps a float")
        if len(x.shape) != 2 or len(y.shape) != 2:
            raise Exception("y and y must be a vector of shape m * 1.")
        if x.shape[1] != 1 or y.shape[1] != 1 or x.shape[0] != y.shape[0]:
            raise Exception("x and y must be a vector of shape m * 1.")

        m, n = x.shape
        y_hat = self.log_predict_(x)
        e = y_hat - y
        X = np.c_[np.ones((m, 1)), x]
        return (np.transpose(X) @ e) / m


    def gradient_descent(self, x, y):
        """
        Args:
            x: has to be an numpy.ndarray, a matrix of shape m * n.
            y: has to be an numpy.ndarray, a vector of shape m * 1.
        Returns:
            new_theta: numpy.array, a vector of dimension (number of features + 1, 1).
        Raises:
            This function should not raise any Exeption.
        """
        for _ in range(self.iter):
            y_hat = self.log_predict_(x)
            
            # On enregistre la loss pour plus tard voir son evolution au cours de l'entrainement
            self.historique.append(self.log_loss_(y, y_hat))

            grad = self.log_gradient(x, y)
            self.theta -= self.alpha * grad


    def stochastic_gradient_descent(self, x, y):
        """
        Args:
            x: has to be an numpy.ndarray, a matrix of shape m * n.
            y: has to be an numpy.ndarray, a vector of shape m * 1.
        Returns:
            new_theta: numpy.array, a vector of dimension (number of features + 1, 1).
        Raises:
            This function should not raise any Exeption.
        """


    def miniBatch_gradient_descent(self, x, y):
        """
        Args:
            x: has to be an numpy.ndarray, a matrix of shape m * n.
            y: has to be an numpy.ndarray, a vector of shape m * 1.
        Returns:
            new_theta: numpy.array, a vector of dimension (number of features + 1, 1).
        Raises:
            This function should not raise any Exeption.
        """
        # a voir je sais pas si sa marche
        m, n = x.shape
        nb_batch = m / self.batch_size
        x_batch = np.array_split(x, nb_batch)
        y_batch = np.array_split(y, nb_batch)
        for _ in range(self.iter):
            #split des data en mini batch
            for i in range(nb_batch):
                y_hat = self.log_predict_(x_batch[i])
            
                # On enregistre la loss pour plus tard voir son evolution au cours de l'entrainement
                self.historique.append(self.log_loss_(y_batch[i], y_hat))

                grad = self.log_gradient(x_batch[i], y_batch[i])
                self.theta -= self.alpha * grad


    def fit_(self, x, y):
        """
        Args:
            x: has to be an numpy.ndarray, a matrix of shape m * n.
            y: has to be an numpy.ndarray, a vector of shape m * 1.
        Returns:
            The gradient as a numpy.ndarray, a vector of shape n * 1.
        Raises:
            This function raise Exeption on any error.
        """
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise Exception("x and y must be a numpy.ndarray, and eps a float")
        if len(x.shape) != 2 or len(y.shape) != 2:
            raise Exception("y and y must be a vector of shape m * 1.")
        if x.shape[1] != 1 or y.shape[1] != 1 or x.shape[0] != y.shape[0]:
            raise Exception("x and y must be a vector of shape m * 1.")

        # on definie l'algo dans l'init:
        # on a 3 choix possible default (GD ou GD par lot), SGD et miniBatch_GD
        # A voir comment on gere les differentes optimization (juste adam)
        self.algo(x, y)
