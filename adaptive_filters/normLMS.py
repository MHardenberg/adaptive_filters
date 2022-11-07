import numpy as np


class normLMS:
    ''' normalized LMS algorith class.
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
vectorize_most_recent(most_recent_signal, filter_order)

compute_correction(self, signal_vectorized, error)

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

    @staticmethod
    def vectorize_most_recent(most_recent_signal, filter_order):
        ''' Returns vectorized signal as
        $$ \mathbf{x}[k] = \left[ \begin{matrix} x[k] \\ x[k-1] \\ \vdots \\ x[k-N+1] \end{matrix} \right] $$'''

        index = len(most_recent_signal)
        if index - filter_order < 0:
            raise Exception(
                f"Time step lower than filter_order. \n len(most_recent_signal) = {len(most_recent_signal)},"
                f" filter_order = {filter_order}. \n No data available beyond index = 0")
        return np.flip(most_recent_signal[index - filter_order: index])

    def compute_correction(self, signal_vectorized, error):
        return 2 * self.heuristic_step_size /\
               (signal_vectorized @ signal_vectorized + self.zero_div_guard) * error * signal_vectorized

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
