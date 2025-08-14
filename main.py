"""Main runner for Carla Drowsiness Detection with multiprocessing."""

import multiprocessing
import time
from typing import Tuple
from src.feature_extraction.camera_features import camera_feature_extraction
from src.carla_api import carla_main
from src.feature_extraction.sensor_fusion import data_association, metrics_calculation
from src.Driver_assistance_bot.bot import Bot, load_json, load_toml


def metrics_worker(
    shared_data: dict[Tuple[Tuple]],
    prompts: dict,
    schema: dict,
    model_id: str,
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

    adas_bot = Bot(model_id, prompts, schema)

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
            print(f"Metrics: {metrics}")
            output = adas_bot.invoke(metrics)
            print("Bot output:", output)

        except Exception as e:
            print(f"Metrics/Bot processing error: {e}")

        # time.sleep(interval)


def main(prompt: dict, schema: dict, model_id: str):
    # Initialize shared data
    manager = multiprocessing.Manager()
    shared_data = manager.dict(
        {
            "camera_frames": manager.list(),
            "vehicle_data": manager.list(),
        }
    )

    # Start child processes
    carla_process = multiprocessing.Process(target=carla_main, args=(shared_data,))
    camera_process = multiprocessing.Process(
        target=camera_feature_extraction, args=(shared_data,)
    )
    metrics_process = multiprocessing.Process(
        target=metrics_worker, args=(shared_data, prompt, schema, model_id, 30, 30)
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
    prompt_ = load_toml(
        r"C:\Users\pasupuleti\Desktop\carla-drowsiness-detection\src\Driver_assistance_bot\configs\prompt.toml"
    )
    schema_ = load_json(
        r"C:\Users\pasupuleti\Desktop\carla-drowsiness-detection\src\Driver_assistance_bot\configs\schema.json"
    )
    MODEL_ID = "llama3.1:8b"
    main(prompt_, schema_, MODEL_ID)
