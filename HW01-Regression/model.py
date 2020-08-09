import numpy as np


class LinearRegression:

    def __init__(self, *, penalty='None', max_iter=1000, tol=0.001, learning_rate='Adagrad', eta0=0.01, w=None):
        """

        Args:
            penalty (string): {'None', 'l2'}, default='None'
                Defaults to l2 which is the standard regularizer for linear models.
            max_iter (int, default=1000): The maximum number of passes over the training data.
            tol (float, default=0.001): The stopping criterion
            learning_rate (string): {'constant', 'Adagrad'}
                'constant': eta = eta0
            eta0 (double, default=0.01): initial learning rate.
            w (ndarray): shape (n_features,)
        """
        self.__penalty = penalty
        self.__max_iter = max_iter
        self.__tol = tol
        self.__learning_rate = learning_rate
        self.__eta0 = eta0
        self.__weight = w

        self.__eta = eta0
        self.__loss = None
        self.__grad = None

        self.__loss_trend = []
        self.__weight_trend = []
        self.__grad_trend = []

        self.__train_x = None
        self.__train_y = None

    def fit(self, X, y):
        """Fit linear model.

        Args:
            X (ndarray): shape (n_samples, n_features)
                Training data
            y (ndarray): shape (n_samples,)
                Target values.
        """
        self.__train_x = np.c_[np.ones(X.shape[0]), X]
        self.__train_y = y.copy()

        self.__reset_trend()
        if self.__weight is None:
            self.__weight = np.zeros(self.__train_x.shape[1])

        grad_square_sum = 0
        eps = 0.0000000001
        i: int
        for i in range(self.__max_iter):
            grad_ = self.__cal_grad()
            if self.__learning_rate == 'Adagrad':
                grad_square_sum += grad_ ** 2
                self.__eta = self.__eta0 / np.sqrt(grad_square_sum + eps)
            next_weight = self.__weight - self.__eta * grad_
            self.__weight = next_weight
            self.__loss_trend.append(self.__cal_loss())
            self.__weight_trend.append(self.__weight)
            self.__grad_trend.append(grad_)

        return self.__weight

    def predict(self, X):
        """

        Args:
            X (ndarray): shape (n_samples, n_features)

        Returns:
            ndarray: Returns predicted values.
        """
        return np.c_[np.ones(X.shape[0]), X] @ self.__weight

    def score(self, X, y):
        """

        Args:
            X (ndarray): shape (n_samples, n_features):
            y (ndarray): shape (n_samples,)

        Returns:
            float: Return the coefficient of determination R^2 of the prediction.
                The coefficient R^2 is defined as (1 - u/v),
                where u is the residual sum of squares ((y_true - y_pred) ** 2).sum() and
                v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum().
                The best possible score is 1.0 and it can be negative
                (because the model can be arbitrarily worse).
                A constant model that always predicts the expected value of y,
                disregarding the input features, would get a R^2 score of 0.0.
        """
        err = y - np.c_[np.ones(X.shape[0]), X] @ self.__weight
        u = err @ err
        dif = y - y.mean()
        v = dif @ dif
        return 1 - u / v

    def set_params(self, **params):
        """Set the parameters of this estimator.

        Args:
            **params (dict): Estimator parameters.

        """
        if 'penalty' in params:
            self.__penalty = params['penalty']
        if 'max_iter' in params:
            self.__max_iter = params['max_iter']
        if 'tol' in params:
            self.__tol = params['tol']
        if 'learning_rate' in params:
            self.__learning_rate = params['learning_rate']
        if 'eat0' in params:
            self.__eta0 = params['eta0']
        if 'w' in params:
            self.__weight = params['w']

    def loss_trend(self):
        return self.__loss_trend

    def weight_trend(self):
        return self.__weight_trend

    def __reset_trend(self):
        self.__loss = []
        self.__weight_trend = []
        self.__grad_trend = []

    def __cal_loss(self):
        _err = self.__train_y - self.__train_x @ self.__weight
        self.__loss = np.sqrt(_err @ _err / self.__train_x.shape[0])
        return self.__loss

    def __cal_grad(self):
        if self.__penalty == 'None':
            self.__grad = 2 * np.dot(self.__train_x.T, np.dot(self.__train_x, self.__weight) - self.__train_y)
#             self.__grad = -2 * self.__train_y @ self.__train_x + \
#                           2 * self.__weight @ self.__train_x.T @ self.__train_x
        return self.__grad

    def test(self):
        print('loss: ', self.__cal_loss())
        print('grad: ', self.__cal_grad())
        print('w', self.__weight)


if __name__ == '__main__':
    train_x = np.array([[1, 0.99], [2, 1.99]])
    train_y = np.array([3, 5])
    model = LinearRegression(max_iter=3000)
    model.fit(train_x, train_y)
    model.test()
    print('score: ', model.score(train_x, train_y))
    test_x = np.array([[3, 3], [4, 4]])
    test_y = model.predict(test_x)
    print('test_y: ', test_y)
