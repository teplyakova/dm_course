import numpy as np
import copy
from classification_report import classification_report


class AdaBoostClassifier:
    def __init__(self, base_estimator=None, n_estimators=50):
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.estimators_ = []
        self.estimator_weights_ = []
        self.sample_weights = None
        self.feature_importances_ = None

    def boost_(self, iboost, X, y, sample_weight):
        new_estimator = copy.copy(self.base_estimator)
        new_estimator.fit(X, y, sample_weight)
        y_pred = new_estimator.predict(X)

        err = np.sum(sample_weight[y_pred != y]) / (np.sum(sample_weight))
        if err == 0:
            return new_estimator, 1.0, None, err

        if iboost == 0:
            b = 1.0
        else:
            b = 0.5 * np.log((1 - err) / err)
        sample_weight = sample_weight * np.exp(b * (y_pred != y))
        sample_weight = sample_weight / np.sum(sample_weight)
        return new_estimator, b, sample_weight, err

    def fit(self, X, y):
        # 2 classes: -1 and 1
        self.sample_weights = np.full(y.shape[0], 1. / y.shape[0])
        self.feature_importances_ = np.zeros(X.shape[1])

        for iboost in xrange(self.n_estimators):
            estimator, b, sample_weight, err = self.boost_(iboost, X, y, self.sample_weights)
            self.estimators_.append(estimator)
            self.estimator_weights_.append(b)
            self.feature_importances_ += b * estimator.feature_importances_
            if sample_weight is None:
                break
            self.sample_weights = sample_weight

        return self

    def staged_predict(self, X):
        final_predictions = np.zeros((self.n_estimators, X.shape[0]))
        for i, estimator in enumerate(self.estimators_):
            stage_prediction = estimator.predict(X)
            if i == 0:
                final_predictions[i] = self.estimator_weights_[i] * stage_prediction
            else:
                final_predictions[i] = final_predictions[i-1] + self.estimator_weights_[i] * stage_prediction

        # Early stopping
        pred = final_predictions[len(self.estimators_)-1]
        for i in xrange(len(self.estimators_), self.n_estimators):
            final_predictions[i] = pred
        return np.sign(final_predictions)

    def staged_f1_score(self, X, y):
        predictions = self.staged_predict(X)
        scores = np.zeros(predictions.shape[0])
        for i in xrange(predictions.shape[0]):
            scores[i] = classification_report(y, predictions[i])
        return scores

    def predict(self, X):
        final_prediction = np.zeros(X.shape[0])
        for i, estimator in enumerate(self.estimators_):
            pred = estimator.predict(X)
            final_prediction += self.estimator_weights_[i] * pred
        return np.sign(final_prediction)