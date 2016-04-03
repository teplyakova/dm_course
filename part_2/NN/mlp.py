import numpy as np
import loss


class MultiLayerPerceptron:
    def __init__(self, input_dim, layer_dim, layer_activation, loss=loss.MSELoss(), max_iter=500,
                 mu=0.1, batch=False, batch_size=100, stop_ratio=0.95, tol=1e-6, iter_before_check=10):
        self.mu = mu
        self.max_iter = max_iter
        self.loss = loss
        self.batch = batch
        self.batch_size = batch_size
        self.stop_ratio = stop_ratio
        self.tol = tol
        self.iter_before_check = iter_before_check
        self.layer_activation = layer_activation

        self.W = []
        self.bias = []

        self.cv = False
        self.cost = []
        self.cv_cost = []

        self.layer_input = [None] * len(layer_dim)
        self.layer_derivatives = [None] * len(layer_dim)
        self.delta = [None] * len(layer_dim)
        self.W_grad = [None] * len(layer_dim)

        # initialize weights
        prev = input_dim
        for dim in layer_dim:
            self.W.append(np.random.randn(dim, prev))
            self.bias.append(np.random.randn(dim))
            prev = dim

    def forward_propagation(self, x):
        tmp = x
        for i, W in enumerate(self.W):
            self.layer_input[i] = tmp
            z = np.dot(W, tmp) + self.bias[i]
            tmp = self.layer_activation[i].value(z)
            self.layer_derivatives[i] = self.layer_activation[i].derivative(z)
        return tmp

    def back_propagation(self, t, y):
        # output layer
        self.delta[-1] = self.loss.derivative(t, y) * self.layer_derivatives[-1]

        # other layers
        for i in reversed(xrange(len(self.W) - 1)):
            self.delta[i] = np.dot(self.W[i + 1].T, self.delta[i + 1]) * self.layer_derivatives[i]

        # count gradient for weights
        for l in xrange(len(self.W)):
            self.W_grad[l] = np.asarray(np.dot(np.matrix(self.layer_input[l]).T, np.matrix(self.delta[l])).T)

    def get_class(self, y):
        return np.argmax(y, axis=1)

    def single_step(self, x, t):
        y = self.forward_propagation(x)
        self.back_propagation(t, y)

        # make gradient step
        for i in xrange(len(self.W)):
            self.W[i] -= self.mu * self.W_grad[i]
            self.bias[i] -= self.mu * self.delta[i]

    def batch_step(self, X_batch, t_batch):
        y = self.forward_propagation(X_batch[0])
        self.back_propagation(t_batch[0], y)
        sum_delta = []
        sum_weight = []
        for i in xrange(len(self.delta)):
            sum_delta.append(np.copy(self.delta[i]))
            sum_weight.append(np.copy(self.W_grad[i]))

        for i in xrange(1, X_batch.shape[0]):
            y = self.forward_propagation(X_batch[i])
            self.back_propagation(t_batch[i], y)
            for j in xrange(len(self.delta)):
                sum_delta[j] += self.delta[j]
                sum_weight[j] += self.W_grad[j]

        # make gradient step
        for i in xrange(len(sum_delta)):
            self.W[i] -= self.mu * sum_weight[i] / X_batch.shape[0]
            self.bias[i] -= self.mu * sum_delta[i] / X_batch.shape[0]

    def fit(self, X_train, y_train, X_cv=None, y_cv=None):
        if X_cv is not None:
            self.cv = True

        for i in xrange(self.max_iter):
            if self.batch:
                idx = np.random.permutation(X_train.shape[0])
                idx_batch = range(0, len(idx), self.batch_size)
                for i_batch in idx_batch:
                    X_batch = X_train[idx[i_batch:(i_batch + self.batch_size)]]
                    t_batch = y_train[idx[i_batch:(i_batch + self.batch_size)]]
                    self.batch_step(X_batch, t_batch)
            else:
                idx = np.random.permutation(X_train.shape[0])
                for j in xrange(y_train.shape[0]):
                    self.single_step(X_train[idx[j]], y_train[idx[j]])

            print 'Iteration {i}'.format(i=i)
            # cv
            y_pred = self.predict(X_train)
            train_cost = self.loss.count(y_train, y_pred) / X_train.shape[0]
            print 'Cost on train is {c}'.format(c=train_cost)
            self.cost.append(train_cost)

            if self.cv:
                y_pred_cv = self.predict(X_cv)
                val_cost = self.loss.count(y_cv, y_pred_cv) / X_cv.shape[0]
                print 'Cost on val is {c}'.format(c=val_cost)
                print
                self.cv_cost.append(val_cost)

            if i > self.iter_before_check:
                if self.cv:
                    if self.stop_ratio * self.cv_cost[-1] > min(self.cv_cost):
                        print 'Stop at {i} iter (cv)'.format(i=i)
                        break
                else:
                    if self.stop_ratio * self.cost[-1] > min(self.cost):
                        print 'Stop at {i} iter (cost)'.format(i=i)
                        break

                if len(self.cost) > 1 and np.fabs(self.cost[-1] - self.cost[-2]) < self.tol:
                    print 'Stop at {i} iter (tol)'.format(i=i)
                    break

    def predict(self, X):
        res = []
        for i in xrange(X.shape[0]):
            res.append(self.forward_propagation(X[i]))
        return np.array(res)
