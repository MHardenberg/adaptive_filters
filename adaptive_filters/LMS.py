import numpy as np


class LMS:
    ''' LMS algorithm class.
    Signature:
normLMS(
    signal,
    desired_signal,
    weights,
    **kwargs
)
Docstring:

Parameters
-------
signal : numpy.ndarray
desired_signal : numpy.ndarray
weights : numpy.ndarray
step_size : float

Methods
-------

compute_correction(self, signal_vectorized, error)

Inherited from AdaptiveFilter:
vectorize_most_recent(most_recent_signal, filter_order)

Inherited from AdaptiveFilter:
adapt_weights()
    Runs algorithm through all data points

-------'''

    def __init__(self, signal, desired_signal, weights, step_size):
        self.signal: np.ndarray = signal
        self.desired_signal: np.ndarray = desired_signal
        self.samples: int = len(signal)

        self.filter_order: int = len(weights)
        self.weights: np.ndarray = np.zeros((self.filter_order, self.samples))
        self.weights[:, 0] = weights

        self.step_size: float = step_size

        self.estimate: np.ndarray = np.zeros(signal.shape)
        self.error: np.ndarray = np.zeros(self.samples - self.filter_order)

    def compute_correction(self, signal_vectorized, error):
        return 2 * self.step_size * signal_vectorized * error

    @staticmethod
    def vectorize_most_recent(most_recent_signal, filter_order):
        index = len(most_recent_signal)
        if index - filter_order < 0:
            raise Exception(
                f"Time step lower than filter_order. \n len(most_recent_signal) = {len(most_recent_signal)},"
                f" filter_order = {filter_order}. \n No data available beyond index = 0")
        return np.flip(most_recent_signal[index - filter_order: index])

    def adapt_weights(self):
        for n in range(self.filter_order, self.samples):
            signal_vectorized = self.vectorize_most_recent(self.signal[:n + 1], self.filter_order)

            # compute current estimate and error
            self.estimate[n] = self.weights[:, n - 1] @ signal_vectorized
            self.error[n - self.filter_order] = self.desired_signal[n] - self.estimate[n]

            # compute lms correction of weights
            correction = self.compute_correction(signal_vectorized, self.error[n - self.filter_order])

            # adapt weights
            self.weights[:, n] = self.weights[:, n - 1] + correction

        return self.weights[:, n]
