import numpy as np
import matplotlib.pyplot as plt
from LMS_ecg import LMS


def compute_desired_signal(signal):
    desired_signal = np.zeros(signal.shape)
    for k, sample in enumerate(signal):
        desired_signal[k] = signal[k] - .25 * desired_signal[k - 1] * (k > 0) + .5 * desired_signal[k - 2] * (k > 1)
        # boolean terms to disallow earlier than 0 samples. These are assumed to be 0.

    return desired_signal


def generate_test_signal(samples):
    signal = np.random.rand(samples)
    desired_signal = compute_desired_signal(signal)
    return signal, desired_signal


def run_test(runs=50, samples=10000, filter_order=5, step_size=1e-3, signal_desired_tuple=None):
    if signal_desired_tuple is not None:
        sum_of_squared_error = np.zeros(len(signal_desired_tuple[0]) - filter_order)
    else:
        sum_of_squared_error = np.zeros(samples - filter_order)

    for run in range(runs):
        if signal_desired_tuple is not None:
            x, d = signal_desired_tuple
        else:
            x, d = generate_test_signal(samples)

        weights = np.zeros(filter_order)
        adaptive_filter = LMS(x, d, weights, step_size)
        adaptive_filter.adapt_weights()

        sum_of_squared_error += adaptive_filter.error**2

    mean_squared_error = sum_of_squared_error / runs

    plt.plot(mean_squared_error)
    plt.show()


if __name__ == '__main__':
    run_test()
