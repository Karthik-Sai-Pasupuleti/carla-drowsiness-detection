"""This module uses the mediapipe to predict the facial landmarks then calculates the
drowsiness metrics like PERCLOS, blink rate, yawning rate.
"""

# pylint: disable=no-member

import time
import mediapipe as mp
import cv2
import numpy as np

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


if __name__ == "__main__":
    # dummy data to run the script
    shared_data = {"camera_frames": []}
    camera_feature_extraction(shared_data)
