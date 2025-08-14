"""_summary_"""

import numpy as np
import cv2

from feature_extraction.drowsiness_inference import *
from feature_extraction.vehicle_features import *
from Driver_assistance_bot.bot import Bot


def main(bot_, theta_: np.array, sampling_freq: float, lane_position: np.array):
    filtered_theta = low_pass_filter(
        theta=theta_, cutoff_freq_hz=2, filter_order=2, sampling_rate=sampling_freq
    )

    entropy = approx_entropy(time_series=filtered_theta, run_length=2)
    steering_reversals_rate = steering_reversals(filtered_theta=filtered_theta)
    sdlp = lane_position_std_dev(lane_position)

    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30

    extractor = CameraBasedFeatureExtractor(fps)
    last_metrics = None  # store last known metrics

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        extractor.process_frame(frame)
        metrics = extractor.get_metrics()

        # Update last_metrics only if new metrics are returned
        if metrics:
            print("metrics: ", metrics)
            last_metrics = metrics

            llm_input = {
            "perclos": last_metrics["perclos"],
            "blink_rate": last_metrics["blink_frequency"],
            "yawn_freq": last_metrics["yawn_frequency"],
            "sdlp": sdlp,
            "steering_entropy": entropy,
            "steering_reversal_rate": steering_reversals_rate,
            }

            output = bot_.invoke(llm_input)
            print(output)



if __name__ == "__main__":
    bot_  = Bot()
    print('bot created')
    theta = np.random.rand(30)
    lp = np.random.rand(30)
    main(bot_, theta, 30, lp)

