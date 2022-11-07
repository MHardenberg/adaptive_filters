import numpy as np
from adaptive_filters.AdaptiveFilter import AdaptiveFilter

class normLMS(AdaptiveFilter):
    ''' normalized LMS algorithm class.
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

kwargs:
mu_heuristic : float
    Heuristic step size for ensuring under-damped conditions. Default = .05
psi : float
    Division by zero guard. Default = .001

Methods
-------
compute_correction(self, signal_vectorized, error)

Inherited from AdaptiveFilter:
vectorize_most_recent(most_recent_signal, filter_order)

Inherited from AdaptiveFilter:
adapt_weights()
    Runs algorithm through all data points

-------'''

    def __init__(self, signal, desired_signal, weights, **kwargs):
        self.signal: np.ndarray = signal
        self.desired_signal: np.ndarray = desired_signal
        self.samples: int = len(signal)

        self.filter_order: int = len(weights)
        self.weights: np.ndarray = np.zeros((self.filter_order, self.samples))
        self.weights[:, 0] = weights

        if 'mu_heuristic' in kwargs:
            self.heuristic_step_size = kwargs['mu_heuristic']
        else:
            self.heuristic_step_size = .05

        if 'psi' in kwargs:
            self.zero_div_guard : float = kwargs['psi']
        else:
            self.zero_div_guard : float = .001

        self.estimate: np.ndarray = np.zeros(signal.shape)
        self.error: np.ndarray = np.zeros(self.samples - self.filter_order)


    def compute_correction(self, signal_vectorized, error):
        return 2 * self.heuristic_step_size /\
               (signal_vectorized @ signal_vectorized + self.zero_div_guard) * error * signal_vectorized
