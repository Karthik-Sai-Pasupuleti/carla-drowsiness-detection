"""This module uses the mediapipe to predict the facial landmarks then calculates the
drowsiness metrics like PERCLOS, blink rate, yawning rate.
"""

# pylint: disable=no-member

import time
import mediapipe as mp
import cv2
import numpy as np

import PySpin

from .utils import calculate_avg_ear, mouth_aspect_ratio


class CameraBasedFeatureExtractor:
    """Initialize mediapipe model"""

    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process_frame(self, frame: np.ndarray):
        """Predicts facial landmarks and calculates EAR and MAR."""
        results = self.face_mesh.process(frame)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            landmark_array = np.array([[lm.x, lm.y] for lm in landmarks.landmark])
            # Calculate EAR
            ear = calculate_avg_ear(landmark_array)
            # Calculate MAR
            mar = mouth_aspect_ratio(landmark_array)

            return ear, mar
        return None, None


def camera_feature_extraction(data: dict):
    """Captures images from the camera and predicts the facial landmarks then calcualte
    the EAR and MAR and update the data which is the globa

    Args:
        data (dict): global dictionary to update the metrics scores
    """

    try:
        extractor = CameraBasedFeatureExtractor()
        cap = cv2.VideoCapture(0)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            current_time = time.time()
            ear, mar = extractor.process_frame(frame)

            data["camera_frames"].append((current_time, ear, mar))

            if mar is not None and ear is not None:
                cv2.putText(
                    frame,
                    f"MAR: {mar}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"EAR: {ear}",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
            cv2.imshow("Drowsiness Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:  # pylint: disable=broad-exception-caught
        print("Error in Camera Feature Module:", e)


# == Pyspin == #


def pyspin_camera_feature_extraction(data):
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()

    if cam_list.GetSize() == 0:
        print("No cameras detected by PySpin.")
        cam_list.Clear()
        system.ReleaseInstance()
        return

    cam = cam_list.GetByIndex(0)

    try:
        cam.Init()
        cam.BeginAcquisition()
        extractor = CameraBasedFeatureExtractor()

        print("Starting acquisition... Press 'q' or ESC to quit.")

        while True:
            try:
                # Acquire image with timeout
                image_result = cam.GetNextImage(1000)  # 1000 ms
            except PySpin.SpinnakerException as e:
                print("Timeout or acquisition error:", e)
                continue

            if image_result.IsIncomplete():
                print(f"Image incomplete with status {image_result.GetImageStatus()}")
                image_result.Release()
                continue

            # Convert image to numpy array
            img_data = image_result.GetNDArray()

            # Ensure RGB format for Mediapipe
            if img_data.ndim == 2:  # Grayscale
                img_color = cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB)
            else:  # Assume BGR from camera
                img_color = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)

            # Process frame
            ear, mar = extractor.process_frame(img_color)
            current_time = time.time()
            data["camera_frames"].append((current_time, ear, mar))

            # Convert back to BGR for OpenCV display
            display_img = cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR)

            if ear is not None and mar is not None:
                cv2.putText(
                    display_img,
                    f"EAR: {ear:.3f}",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    display_img,
                    f"MAR: {mar:.3f}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Drowsiness Detection (PySpin Camera)", display_img)

            key = cv2.waitKey(1) & 0xFF
            image_result.Release()  # Always release after use

            if key in [ord("q"), 27]:  # 'q' or ESC
                break

    except Exception as e:
        print("Error in PySpin Camera Feature Module:", e)

    finally:
        try:
            cam.EndAcquisition()
        except Exception:
            pass
        try:
            cam.DeInit()
        except Exception:
            pass

        cam_list.Clear()
        system.ReleaseInstance()
        cv2.destroyAllWindows()
        print("Camera and system released successfully.")


# == Pyspin == #

if __name__ == "__main__":
    # dummy data to run the script
    shared_data = {"camera_frames": []}
    camera_feature_extraction(shared_data)
