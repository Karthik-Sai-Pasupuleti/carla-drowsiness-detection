"""
CSV Synchronization Script: Matches camera images with closest driving data timestamps.

Assumes:
- Camera CSV contains: 'timestamp' (UNIX float), 'image_filename'
- Driving CSV contains: 'timestamp' (UNIX float), 'steering', 'offset'

Outputs:
- Synced CSV with columns: timestamp, timestamp_float, ir_filename, steering_angle, lane_offset

Author: Ghulam Rasool
"""

import pandas as pd

# -------- STEP 1: LOAD CSVs AND CONVERT TIMESTAMPS --------

def load_csv_data(camera_csv_path, driving_csv_path):
    """
    Loads camera and driving CSVs and converts timestamp columns to datetime objects.

    Args:
        camera_csv_path (str): Path to CSV file with image timestamps and filenames.
        driving_csv_path (str): Path to CSV file with steering and lane offset data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrames for camera and driving data.
    """
    cam_df = pd.read_csv(camera_csv_path)
    drive_df = pd.read_csv(driving_csv_path)

    # Convert float UNIX timestamps to pandas datetime format
    cam_df['timestamp'] = pd.to_datetime(cam_df['timestamp'], unit='s')
    drive_df['timestamp'] = pd.to_datetime(drive_df['timestamp'], unit='s')

    return cam_df, drive_df


# -------- STEP 2: SYNC (Reference = Steering/Lane data) --------

def sync_data(cam_df, drive_df, output_path):
    """
    Synchronizes driving data with the closest corresponding camera image timestamps.

    For each row in `drive_df`, finds the closest timestamp in `cam_df`
    and builds a new synced row containing:
    - timestamp from driving data
    - matched image filename from camera data
    - steering angle and lane offset

    Args:
        cam_df (pd.DataFrame): DataFrame with 'timestamp' and 'image_filename'.
        drive_df (pd.DataFrame): DataFrame with 'timestamp', 'steering', and 'offset'.
        output_path (str): Output file path for the synced CSV.

    Returns:
        None
    """
    synced_rows = []

    for _, drive_row in drive_df.iterrows():
        drive_ts = drive_row['timestamp']

        # Find camera frame with the closest timestamp
        closest_cam = cam_df.iloc[(cam_df['timestamp'] - drive_ts).abs().argmin()]

        synced_rows.append({
            "timestamp": drive_ts,
            "timestamp_float": drive_ts.timestamp(),  # High-precision float timestamp
            "ir_filename": closest_cam['image_filename'],
            "steering_angle": drive_row['steering'],
            "lane_offset": drive_row['offset']
        })

    # Convert to DataFrame and export
    df_out = pd.DataFrame(synced_rows)
    df_out.to_csv(
        output_path,
        index=False,
        date_format='%Y-%m-%d %H:%M:%S.%f',
        float_format='%.9f'
    )

    print(f"[Sync] Saved {len(df_out)} synced rows to {output_path}")


# -------- RUNNER --------

def run_csv_sync(camera_csv_path, driving_csv_path, output_csv_path):
    """
    Runs the full sync pipeline:
    1. Loads camera and driving data
    2. Synchronizes data
    3. Saves to output CSV

    Args:
        camera_csv_path (str): Path to camera metadata CSV.
        driving_csv_path (str): Path to steering/lane CSV.
        output_csv_path (str): Destination path for synced output CSV.
    """
    cam_df, drive_df = load_csv_data(camera_csv_path, driving_csv_path)
    sync_data(cam_df, drive_df, output_csv_path)


# -------- ENTRY POINT --------

if __name__ == "__main__":
    run_csv_sync(
        camera_csv_path=r"C:\Users\hussa\OneDrive\Desktop\Projects\LLM Project\image_metadata.csv",  # CSV with columns: timestamp, image_filename
        driving_csv_path=r"C:\Users\hussa\OneDrive\Desktop\Projects\LLM Project\data_capture.csv",  # CSV with columns: timestamp, steering, offset
        output_csv_path="synced_output.csv"  # Output combined synced CSV
    )
