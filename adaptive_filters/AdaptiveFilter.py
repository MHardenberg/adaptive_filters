import numpy as np


class AdaptiveFilter:
    def __init__(self, signal, desired_signal, weights):
        self.signal: np.ndarray = signal
        self.desired_signal: np.ndarray = desired_signal
        self.samples: int = len(signal)

        self.filter_order: int = len(weights)
        self.weights: np.ndarray = np.zeros((self.filter_order, self.samples))
        self.weights[:, 0] = weights

        self.estimate: np.ndarray = np.zeros(signal.shape)
        self.error: np.ndarray = np.zeros(self.samples - self.filter_order)

    @staticmethod
    def vectorize_most_recent(most_recent_signal, filter_order):
        index = len(most_recent_signal)
        if index - filter_order < 0:
            raise Exception(
                f"Time step lower than filter_order. \n len(most_recent_signal) = {len(most_recent_signal)},"
                f" filter_order = {filter_order}. \n No data available beyond index = 0")
        return np.flip(most_recent_signal[index - filter_order: index])

    def compute_correction(self, *args):
        return 0

    def adapt_weights(self):
        for n in range(self.filter_order, self.samples - 1):
            signal_vectorized = self.vectorize_most_recent(self.signal[:n + 1], self.filter_order)

            # compute current estimate and error
            self.estimate[n] = self.weights[:, n] @ signal_vectorized
            self.error[n - self.filter_order] = self.desired_signal[n] - self.estimate[n]

            # compute lms correction of weights
            correction = self.compute_correction(signal_vectorized, self.error[n - self.filter_order])

            # adapt weights
            self.weights[:, n + 1] = self.weights[:, n] + correction

        return self.weights[:, n]
