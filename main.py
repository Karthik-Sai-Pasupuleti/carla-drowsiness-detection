"""Main runner for Carla Drowsiness Detection with multiprocessing."""

import argparse
from pathlib import Path
import multiprocessing
import time
from typing import Tuple
import evdev
from evdev import ecodes, InputDevice


from src.feature_extraction.camera_features import (
    camera_feature_extraction,
    pyspin_camera_feature_extraction,
)

from src.carla_api.manual_control_keyboard import carla_keyboard
from src.carla_api.manual_control_steering_wheel import carla_steering_wheel
from src.feature_extraction.sensor_fusion import data_association, metrics_calculation
from src.Driver_assistance_bot.basebot import DriverAssistanceBot, BotConfig
from src.Driver_assistance_bot.utils import load_toml, load_json


def steering_auto_centering(val: int = 35535):
    device = evdev.list_devices()[0]
    evtdev = InputDevice(device)
    # val \in [0,65535]
    evtdev.write(ecodes.EV_FF, ecodes.FF_RUMBLE, val)


def metrics_worker(
    shared_data: dict[Tuple[Tuple]],
    config: BotConfig,
    time_window: int,
    interval: int,
):
    """Continuously calculate metrics and invoke ADAS bot.

    Args:
        shared_data (dict[Tuple[Tuple]]): dictionary with vehicle and camera data with timestamps.
        time_window (int): duration of previous timestamps to calculate metrics.
        prompts (dict): template with system and user prompt.
        schema (dict): structured output schema.
        model_id (str): model id to initialize llm.
        interval (int): time to next iteration.
    """

    adas_bot = DriverAssistanceBot(config)

    while True:
        try:
            current_time = time.time()
            # Get last time_window seconds of data
            ear, mar, lane_position, steering_angle = data_association(
                shared_data, current_time, time_window
            )

            if len(ear) == 0 and len(mar) == 0:
                # No new data to process
                # time.sleep(interval)
                continue

            metrics = metrics_calculation(
                ear,
                mar,
                lane_position,
                steering_angle,
                ear_threshold=0.2,
                con_frames=3,
                mar_threshold=0.5,
                window=time_window,
            )
            time.sleep(30)
            print(f"Metrics: {metrics}")
            output = adas_bot.invoke(metrics)
            print("Bot output:", output)

        except Exception as e:
            print(f"Metrics/Bot processing error: {e}")


def main(config: BotConfig, control: str):
    # Initialize shared data
    manager = multiprocessing.Manager()
    shared_data = manager.dict(
        {
            "camera_frames": manager.list(),
            "vehicle_data": manager.list(),
        }
    )

    # Select Carla control method
    if control == "keyboard":
        carla_target = carla_keyboard
    elif control == "steering":
        carla_target = carla_steering_wheel
    else:
        raise ValueError("Invalid control option. Choose 'keyboard' or 'steering'.")

    # Start child processes
    carla_process = multiprocessing.Process(target=carla_target, args=(shared_data,))
    camera_process = multiprocessing.Process(
        target=pyspin_camera_feature_extraction, args=(shared_data,)
    )
    metrics_process = multiprocessing.Process(
        target=metrics_worker, args=(shared_data, config, 30, 30)
    )

    processes = [carla_process, camera_process, metrics_process]

    for p in processes:
        p.start()

    print("All processes started. Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(1)
            # If either Carla or Camera process dies, terminate all
            if not carla_process.is_alive() or not camera_process.is_alive():
                print("Critical process terminated. Shutting down all processes...")
                break
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Shutting down processes...")
    finally:
        # Terminate all processes
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join()
        print("All processes terminated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Carla Drowsiness Detection Runner")
    parser.add_argument(
        "--control",
        type=str,
        choices=["keyboard", "steering"],
        default="steering",  # default is steering now
        help="Choose Carla control method",
    )
    args = parser.parse_args()

    prompt_file = Path("src") / "Driver_assistance_bot" / "configs" / "prompt.toml"
    schema_file = Path("src") / "Driver_assistance_bot" / "configs" / "schema.json"
    prompt_ = load_toml(prompt_file)
    schema_ = load_json(schema_file)
    MODEL_ID = "phi3:mini"
    config_ = BotConfig(model_id=MODEL_ID, prompts=prompt_, schema=schema_)
    steering_auto_centering(35000)
    main(config_, args.control)
