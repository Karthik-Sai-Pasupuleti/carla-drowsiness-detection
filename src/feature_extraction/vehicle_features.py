"""
This module contains the functions for calculating the vehicle based metrics from
steering wheel and lane positioning.
"""

import math
from typing import Tuple, List
import numpy as np
from scipy.signal import butter, filtfilt

__all__ = [
    "low_pass_filter",
    "approx_entropy",
    "count_reversals",
    "steering_reversals",
    "lane_position_std_dev",
]


def low_pass_filter(
    theta: np.array, cutoff_freq_hz: float, filter_order: float, sampling_rate: float
) -> np.array:
    """Low Pass Filter

    Args:
        theta (np.array): raw steering data

        cutoff_freq_hz (float): low-pass Butterworth filter cutoff frequency (Hz), 2Hz is
        recommended as the optimal parameter for cognitive load based on findings from the
        literature: "A Steering Wheel Reversal Rate Metric for Assessing Effects of Visual
        and Cognitive Secondary Task Load"

        filter_order (float): order of butterworth filter, 2nd order is recommended
        sampling_rate (float): sampling freq in Hz (samples per second)

    Returns:
        np.array: Filtered steering data
    """
    nyquist_freq = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq_hz / nyquist_freq
    b, a = butter(filter_order, normal_cutoff, btype="low", analog=False)
    theta_filtered = filtfilt(b, a, theta)

    return theta_filtered


# === STEERING WHEEL MOVEMENT ANALYSIS ===
# reference paper: [A Steering Wheel Reversal Rate Metric for Assessing Effects of Visual
# and Cognitive Secondary Task Load](https://core.ac.uk/download/pdf/159068039.pdf)


def approx_entropy(time_series: np.array, run_length: int = 2) -> float:
    """Approximate entropy (2sec window) [https://www.mdpi.com/1424-8220/17/3/495]

    Args:
        time_series (np.array): steering movement data
        run_length (int): length of the run data (window with overlapping of the data,
        example x = [1,2,3], if runlength=2 then output will be [[1,2], [2,3]])

    Returns:
        float: regularity (close to 0 : no irregularity, close to 1: irregularity)
    """
    std_dev = np.std(time_series)
    filter_level = 0.2 * std_dev

    def _maxdist(x_i, x_j):
        return max(abs(ua - va) for ua, va in zip(x_i, x_j))

    def _phi(m):
        n = time_series_length - m + 1
        x = [
            [time_series[j] for j in range(i, i + m - 1 + 1)]
            for i in range(time_series_length - m + 1)
        ]
        counts = [
            sum(1 for x_j in x if _maxdist(x_i, x_j) <= filter_level) / n for x_i in x
        ]
        return sum(math.log(c) for c in counts) / n

    time_series_length = len(time_series)

    return abs(_phi(run_length + 1) - _phi(run_length))


def count_reversals(
    theta_vals: np.array, gap: float
) -> Tuple[int, List[Tuple[float, float]]]:
    """calculates steering reversal count

    Args:
        theta_vals (np.array): steering angles
        gap (float): threeshold

    Returns:
        Tuple[int, List[Tuple[float, float]]]: reversal count, list of reversal indices
    """

    k = 0
    Nr = 0
    R = []
    N = len(theta_vals)
    for l in range(1, N):
        if theta_vals[l] - theta_vals[k] >= gap:
            Nr += 1
            R.append((k, l))
            k = l
        elif theta_vals[l] < theta_vals[k]:
            k = l
    return Nr, R


def steering_reversals(filtered_theta: np.array, theta_min: float = 0.1) -> int:
    """calculate the steering wheel reversals of both upward and downward

    Args:
        filtered_theta (np.array): filtered steering wheel data
        theta_min (float): gap size threeshold

    Returns:
        int: reversal count
    """

    # Calculate discrete derivative
    diff_x = np.diff(filtered_theta)
    sign_diff = np.sign(diff_x)

    stationary_points = [0]  # include first index

    for i in range(1, len(sign_diff)):
        if sign_diff[i] != sign_diff[i - 1]:
            stationary_points.append(i)

    stationary_points.append(len(filtered_theta) - 1)  # include last index

    nr_up, _ = count_reversals(filtered_theta, theta_min)
    # To count downward, repeat on negative signal
    nr_down, _ = count_reversals(-filtered_theta, theta_min)

    total_reversals = nr_up + nr_down

    return total_reversals


# === LANE POSITION ===


def lane_position_std_dev(deviations: np.array) -> float:
    """Calculate the standard deviation of lane position deviations.

    Args:
        deviations (np.array): Array of lane position deviations from the lane center.

    Returns:
        float: Standard deviation of the deviations.
    """

    deviations = np.array(deviations)
    std_dev = np.std(deviations, ddof=0)  # population standard deviation
    return std_dev
