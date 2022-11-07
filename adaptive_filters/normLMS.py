import numpy as np
from adaptive_filters.LMS import LMS


class normLMS(LMS):
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

Inherited from LMS:
vectorize_most_recent(most_recent_signal, filter_order)

Inherited from AdaptiveFilter:
adapt_weights()
    Runs algorithm through all data points

-------'''

    def __init__(self, signal, desired_signal, weights, **kwargs):
        super().__init__(signal, desired_signal, weights, step_size=0)

        if 'mu_heuristic' in kwargs:
            self.heuristic_step_size = kwargs['mu_heuristic']
        else:
            self.heuristic_step_size = .05

        if 'psi' in kwargs:
            self.zero_div_guard : float = kwargs['psi']
        else:
            self.zero_div_guard : float = .001

    def compute_correction(self, signal_vectorized, error):
        return 2 * self.heuristic_step_size /\
               (signal_vectorized @ signal_vectorized + self.zero_div_guard) * error * signal_vectorized
