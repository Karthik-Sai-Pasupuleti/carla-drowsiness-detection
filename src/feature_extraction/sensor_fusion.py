"""This module contains the data accociation and drowsiness metrics calculation."""

from typing import Tuple, List
import numpy as np

from .utils import (
    calculate_perclos,
    calculate_blink_frequency,
    calculate_yawn_frequency,
    vehicle_feature_extraction,
)


def data_association(
    data: dict, current_time, time_window=30
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Data association of a time window based on the timestamps of camera and vehicle features.

    Args:
        data (dict): camera and vehicle features with timestamps
        current_time (float|int|str): current time to filter the data
        time_window (float|int|str, optional): calculate the metric scores for last x seconds. Defaults to 30.

    Returns:
        Tuple[np.array[float], np.array[float], np.array[float], np.array[float]]: filtered features
    """

    # Ensure numeric types
    try:
        # print(f"current time: {current_time}, time_window: {time_window}")
        current_time = float(current_time)
        time_window = float(time_window)
    except ValueError as e:
        raise ValueError(
            f"current_time and time_window must be numeric. Got: {current_time}, {time_window}"
        ) from e

    window_start = current_time - time_window

    # CAMERA DATA
    frames = list(data.get("camera_frames", []))  # [(ts, ear, mar), ...]
    filtered_frames = [
        (ts, e, m) for ts, e, m in frames if window_start <= ts <= current_time
    ]
    remaining_frames = [(ts, e, m) for ts, e, m in frames if ts < window_start]
    data["camera_frames"][:] = remaining_frames

    if filtered_frames:
        _, filtered_ear, filtered_mar = zip(*filtered_frames)
    else:
        filtered_ear, filtered_mar = [], []

    # VEHICLE DATA
    veh_frames = list(
        data.get("vehicle_data", [])
    )  # [(ts, steering_angle, lane_position), ...]
    filtered_veh_frames = [
        (ts, sa, lp) for ts, sa, lp in veh_frames if window_start <= ts <= current_time
    ]
    remaining_veh_frames = [
        (ts, sa, lp) for ts, sa, lp in veh_frames if ts < window_start
    ]
    data["vehicle_data"][:] = remaining_veh_frames

    if filtered_veh_frames:
        _, filtered_steering_angle, filtered_lane_position = zip(*filtered_veh_frames)
    else:
        filtered_steering_angle, filtered_lane_position = [], []

    return (
        np.array(filtered_ear),
        np.array(filtered_mar),
        np.array(filtered_lane_position),
        np.array(filtered_steering_angle),
    )


def metrics_calculation(
    ear: List,
    mar: List,
    lane_pos: List,
    steering_angle: List,
    ear_threshold: float,
    con_frames: int,
    mar_threshold: float,
    window: float,
) -> dict:
    """
    Calculate the metric scores.

    Args:
        ear (List): EAR values.
        mar (List): MAR values.
        lane_pos (List): Lane position data.
        steering_angle (List): Steering angle data.
        ear_threshold (float): Threshold to determine eye closure.
        con_frames (int): Consecutive frames to calculate a blink.
        mar_threshold (float): Threshold to determine mouth opening.
        window (float): Time window in seconds.

    Returns:
        dict: Metric scores from vehicle and camera data.
    """
    fps = len(ear) / window if window > 0 else 0

    # Facial feature metrics
    try:
        perclos = calculate_perclos(ear, ear_threshold, con_frames)
        blink_rate = calculate_blink_frequency(ear, ear_threshold, fps)
        yawn_freq = calculate_yawn_frequency(mar, mar_threshold, int(fps * 4), fps)
    except (ValueError, RuntimeError) as e:
        print(f"Facial metrics error: {e}")
        perclos = blink_rate = yawn_freq = None

    # Vehicle feature metrics
    try:
        entropy, steering_rate, sdlp = vehicle_feature_extraction(
            steering_angle, lane_pos, window
        )
    except (ValueError, RuntimeError) as e:
        print(f"Vehicle metrics error: {e}")
        entropy = steering_rate = sdlp = None

    return {
        "perclos": perclos,
        "blink_rate": blink_rate,
        "yawn_freq": yawn_freq,
        "sdlp": sdlp,
        "steering_entropy": entropy,
        "steering_reversal_rate": steering_rate,
    }
