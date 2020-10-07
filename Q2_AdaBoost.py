import numpy as np


class AdaBoost:
    def __init__(self, T=50):
        self.T = T
        self.weak_classifiers = []

    def train(self, X, Y):
        # m is the number of samples, n is number of features
        m, n = X.shape
        # initialize weights
        w = np.ones(m) / m

        for t in range(self.T):
            # initialize a dictionary as weak classifier selector
            best_weak_classifier = {'feature': 0, 'threshold': 0, 'min_error': np.inf,
                                    'is_greater_than_threshold': 1, 'alpha': 0}

            # search for the best weak classifier in all features
            for j in range(n):
                x_samples = X[:, j]

                # sort the samples from lowest value to highest
                thresholds = np.sort(x_samples)

                # search for the best threshold that can minimize the total error
                for threshold in thresholds:

                    # we predict the sample value that is greater than threshold as positive prediction
                    is_greater_than_threshold = 1
                    predictions = np.ones(m)
                    predictions[x_samples < threshold] = -1

                    # sum of weights of misclassified samples to get the total error
                    misclassified_weights = w[Y != predictions]
                    total_error = sum(misclassified_weights)

                    # Since we are minimizing error, if the error rate is higher than 0.5, we flip the sign
                    # (from "greater than threshold" to the "less than threshold") to get the minimum error rate
                    if total_error > 0.5:
                        total_error = 1 - total_error

                        # reset is_greater_than_threshold = -1 means now we mark the sample value that
                        # less than threshold as positive prediction in this weak classifier
                        is_greater_than_threshold = -1

                    if total_error < best_weak_classifier['min_error']:

                        # only keep the best weak classifier's details
                        best_weak_classifier['feature'] = j
                        best_weak_classifier['threshold'] = threshold
                        best_weak_classifier['min_error'] = total_error
                        best_weak_classifier['is_greater_than_threshold'] = is_greater_than_threshold

            # calculate the alpha, add 1e-10 on denominator to avoid dividing by zero
            best_weak_classifier['alpha'] = 1 / 2 * np.log(
                (1.0 - best_weak_classifier['min_error']) / max(best_weak_classifier['min_error'], 1e-10))

            # store the best decision stump into weak classifier list
            self.weak_classifiers.append(best_weak_classifier)

            # get predictions on a single weak classifier
            weak_classifier_predictions = self.weak_classify(X[:, best_weak_classifier['feature']],
                                                             best_weak_classifier['threshold'],
                                                             best_weak_classifier['is_greater_than_threshold'])

            # update the weights
            w = w * np.exp((-1) * float(best_weak_classifier['alpha']) * Y * weak_classifier_predictions)
            # get the normalization factor Z
            z = np.sum(w)
            w = w / z
            print("alpha", t + 1, ': ', best_weak_classifier['alpha'])

    @staticmethod
    def weak_classify(X_column, threshold, is_greater_than_threshold):
        # predict based on single weak classifier
        predictions = np.ones(X_column.shape[0])
        if is_greater_than_threshold == 1:
            predictions[X_column < threshold] = -1
        else:
            predictions[X_column > threshold] = -1

        return predictions

    def predict(self, X):
        prediction = [wcl['alpha'] * self.weak_classify(
            X[:, wcl['feature']], wcl['threshold'], wcl['is_greater_than_threshold']) for wcl in self.weak_classifiers]
        Y = np.sum(prediction, axis=0)
        Y = np.sign(Y)
        return Y
