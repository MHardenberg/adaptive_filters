import numpy as np
from adaptive_filters.AdaptiveFilter import AdaptiveFilter

class LMS(AdaptiveFilter):
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
