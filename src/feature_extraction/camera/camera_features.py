"""This module contains the camera-based feature extraction class for drowsiness
detection. It uses MediaPipe for face landmark detection to calculate Eye Aspect Ratio
(EAR) and blink detection. Using these, PERCLOS (Percentage of Eye Closure) and blink
frequency are calculated.

MediaPipe is chosen for this pipeline over dlib because it has better performance and
accuracy. Please refer to the attached research paper for more details:
https://ieeexplore.ieee.org/document/10039811

Reference:
https://github.com/Pushtogithub23/Eye-Blink-Detection-using-MediaPipe-and-OpenCV
"""

from typing import Union, Optional
import mediapipe as mp
import cv2
import numpy as np


class EyeFeatureExtractor:
    """Class for extracting Eye Aspect Ratio (EAR), tracking blinks, and calculating PERCLOS."""

    def __init__(self):
        """Initialize the MediaPipe Face Mesh solution and feature tracking variables."""

        # landmark indices
        self.landmarks_indices = {
            "right_eye": [33, 159, 158, 133, 153, 145],
            "left_eye": [362, 380, 374, 263, 386, 385],
        }

        # EAR and blink tracking
        self.ear_stats = {
            "min": None,  # to caliberate the EAR threshold minimum value will be updated
            "max": None,  # to caliberate the EAR threshold maximum value will be updated
            "threshold": 0.2,  # article for ear threeshold: https://www.mdpi.com/2313-433X/9/5/91.
            "blink_counter": 0,  # blink count for blink frequency calculation, it will be reseted to 0
            "blinks": 0,  # total blinks detected
            "eyes_closed": False,  # Flag to indicate if eyes are currently closed
        }

        # PERCLOS tracking
        self.perclos_stats = {
            "consec_frames": 0,  # Consecutive frames with closed eyes
            "closed_frames": 0,  # Total frames with closed eyes, it will be updated with the consecutive frames
            "min_consec": 3,  # Minimum consecutive frames to consider eyes closed
        }

    def calculate_ear(self, landmarks: np.ndarray, eye: str) -> float:
        """
        Calculate the Eye Aspect Ratio (EAR) for a given eye.

        The EAR is calculated using the formula:
        EAR = (||p2-p6|| + ||p3-p5||) / (2||p1-p4||)
        where p1, p2, p3, p4, p5, p6 are 2D landmark points.

        Args:
            landmarks (np.ndarray): Array of face landmarks.
            eye (str): Either 'left_eye' or 'right_eye'.

        Returns:
            float: The calculated Eye Aspect Ratio
        """
        indices = self.landmarks_indices[eye]
        a = np.linalg.norm(landmarks[indices[1]] - landmarks[indices[5]])
        b = np.linalg.norm(landmarks[indices[2]] - landmarks[indices[4]])
        c = np.linalg.norm(landmarks[indices[0]] - landmarks[indices[3]])
        return (a + b) / (2.0 * c)

    def calculate_avg_ear(self, landmarks: np.ndarray) -> float:
        """Calculate the average Eye Aspect Ratio (EAR) from the face landmarks.
        refer to the paper for more details: https://ieeexplore.ieee.org/document/10039811

        Args:
            landmarks (np.ndarray): Array of face landmarks.

        Returns:
            float: Average Eye Aspect Ratio.
        """
        left_ear = self.calculate_ear(landmarks, "left_eye")
        right_ear = self.calculate_ear(landmarks, "right_eye")
        return (left_ear + right_ear) / 2.0

    def update_ear_threshold(self, ear: float) -> float:
        """Update the EAR threshold based on the current EAR value.
        this approach is refferred from the paper:
        "https://ieeexplore.ieee.org/document/10039811"

        Args:
            ear (float): Current EAR value.

        Returns:
            float: Updated threshold.
        """
        if self.ear_stats["min"] is None or ear < self.ear_stats["min"]:
            self.ear_stats["min"] = ear
        if self.ear_stats["max"] is None or ear > self.ear_stats["max"]:
            self.ear_stats["max"] = ear
        self.ear_stats["threshold"] = (
            self.ear_stats["min"] + self.ear_stats["max"]
        ) / 2.0
        return self.ear_stats["threshold"]

    def update_blink_count(self, ear: float):
        """Update blink count based on current EAR.

        Args:
            ear (float): Current EAR value.
        """
        threshold = self.ear_stats["threshold"]  # to check if the eyes are closed
        if ear < threshold:
            if not self.ear_stats["eyes_closed"]:
                self.ear_stats["eyes_closed"] = True
        else:
            if self.ear_stats["eyes_closed"]:
                self.ear_stats["blink_counter"] += 1
                self.ear_stats["blinks"] += 1
            self.ear_stats["eyes_closed"] = False

    def update_perclos_counter(self, ear: float):
        """Update the PERCLOS counter based on the EAR value and by checking the number
        of consecutive frames with closed eyes and updating the closed frames count.

        Args:
            ear (float): Current EAR value.
        """
        if ear < self.ear_stats["threshold"]:
            self.perclos_stats["consec_frames"] += 1
        else:
            if self.perclos_stats["consec_frames"] >= self.perclos_stats["min_consec"]:
                # update the closed frame count if it exceeds the minimum consecutive
                # frames
                self.perclos_stats["closed_frames"] += self.perclos_stats[
                    "consec_frames"
                ]
            # Reset the consecutive frames counter after updating the closed frames
            self.perclos_stats["consec_frames"] = 0

    def get_perclos(self, total_frames: int) -> float:
        """Compute the PERCLOS metric.

        Args:
            total_frames (int): Total frames in window.

        Returns:
            float: PERCLOS percentage.
        """
        return (self.perclos_stats["closed_frames"] / total_frames) * 100.0

    def get_blink_frequency(self, total_frames: int, fps: float) -> float:
        """Compute blink frequency.

        Args:
            total_frames (int): Total frames.
            fps (float): Frames per second.

        Returns:
            float: Blink frequency (blinks/min).
        """
        return (self.ear_stats["blink_counter"] * 60) / (total_frames / fps)

    def reset_window_stats(self):
        """Reset counters for new PERCLOS and blink frequency window."""
        self.perclos_stats["closed_frames"] = 0
        self.ear_stats["blink_counter"] = 0


class MouthFeatureExtractor:
    """Class for extracting Mouth Aspect Ratio (MAR)."""

    def __init__(self, fps: float):
        """
        Args:
            fps (float): frames per seconds of the camera.
        """
        self.mar_stats = {
            "consec_frames": 0,
            "threeshold": 0.5,
            "min_consec_frames": 4
            * fps,  # yawning is a quick act of opening and closing the mouth, which lasts for around 4 to 6 s.
            "yawning_count": 0,
            "yawning_count_freq": 0,
            "yawn_detected": False,
        }

    def mouth_aspect_ratio(self, landmarks: np.array) -> float:
        """Calculate the Mouth Aspect Ratio (MAR) from the landmarks.

        Args:
            landmarks (np.array): Array of facial landmarks.

        Returns:
            float: The calculated Mouth Aspect Ratio.
        """
        # Define mouth landmarks indices
        mouth_indices = [78, 81, 13, 311, 308, 402, 14, 178]

        # Calculate distances between the landmarks
        a = np.linalg.norm(
            np.array(landmarks[mouth_indices[1]])
            - np.array(landmarks[mouth_indices[7]])
        )
        b = np.linalg.norm(
            np.array(landmarks[mouth_indices[2]])
            - np.array(landmarks[mouth_indices[6]])
        )
        c = np.linalg.norm(
            np.array(landmarks[mouth_indices[3]])
            - np.array(landmarks[mouth_indices[5]])
        )

        d = np.linalg.norm(
            np.array(landmarks[mouth_indices[0]])
            - np.array(landmarks[mouth_indices[4]])
        )

        return (a + b + c) / (2.0 * d)

    def yawning_detection(self, mar: float):
        """This functions compares the mar value with the threeshold and checks for the
        yawning detection by checking the number of consecutive frames with mouth opened.

        Args:
            mar (float): mar values.
        """
        if mar > self.mar_stats["threeshold"]:
            self.mar_stats["consec_frames"] += 1

            if (
                self.mar_stats["consec_frames"] >= self.mar_stats["min_consec_frames"]
                and self.mar_stats["yawn_detected"] == False
            ):
                self.mar_stats["yawning_count"] += 1
                self.mar_stats["yawn_detected"] = True
                self.mar_stats["yawning_count_freq"] += 1
        else:
            self.mar_stats["consec_frames"] = 0
            self.mar_stats["yawn_detected"] = False

    def get_yawn_frequency(self, total_frames: int, fps: float) -> float:
        """Calculate the yawn frequency in yawns per minute.

        Args:
            total_frames (int): Total number of frames processed in the current window.
            fps (float): Frames per second of the video feed.

        Returns:
            float: Yawn frequency in yawns per minute.
        """

        return (self.mar_stats["yawning_count_freq"] * 60) / (total_frames / fps)

    def reset_yawn_stats(self):
        """Reset yawn counters for the next window."""
        self.mar_stats["yawning_count_freq"] = 0
        self.mar_stats["consec_frames"] = 0
        self.mar_stats["yawn_detected"] = False


class CameraBasedFeatureExtractor:
    """Unified class for extracting drowsiness-related features from camera feed."""

    def __init__(self, fps: float):
        self.fps = fps
        self.eye = EyeFeatureExtractor()
        self.mouth = MouthFeatureExtractor(fps)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.total_frames = 0
        self.perclos_window = int(fps * 30)  # 30 seconds window

    def process_frame(self, frame: np.ndarray):
        """Process a single frame and update internal stats."""
        results = self.face_mesh.process(frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            landmark_array = np.array([[lm.x, lm.y] for lm in landmarks.landmark])

            # Update EAR-based features
            ear = self.eye.calculate_avg_ear(landmark_array)
            # self.eye.update_ear_threshold(ear)
            self.eye.update_blink_count(ear)
            self.eye.update_perclos_counter(ear)

            # Update MAR-based features
            mar = self.mouth.mouth_aspect_ratio(landmark_array)
            self.mouth.yawning_detection(mar)

            self.total_frames += 1

    def get_metrics(self) -> dict:
        """Compute blink/yawn frequency and PERCLOS every 30 seconds."""
        if self.total_frames < self.perclos_window:
            return {}

        blink_freq = self.eye.get_blink_frequency(self.total_frames, self.fps)
        perclos = self.eye.get_perclos(self.total_frames)
        yawn_freq = self.mouth.get_yawn_frequency(self.total_frames, self.fps)

        # Reset for next window
        self.eye.reset_window_stats()
        self.mouth.reset_yawn_stats()
        self.total_frames = 0

        return {
            "blink_frequency": blink_freq,
            "perclos": perclos,
            "yawn_frequency": yawn_freq,
            "blinks": self.eye.ear_stats["blinks"],
        }

def main():
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30

    extractor = CameraBasedFeatureExtractor(fps)
    last_metrics = None  # <-- store last known metrics

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

        # Display last known metrics (if any)
        if last_metrics:
            cv2.putText(
                frame,
                f"Blink Freq: {last_metrics['blink_frequency']:.2f} blinks/min",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"PERCLOS: {last_metrics['perclos']:.2f}%",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Yawn Freq: {last_metrics['yawn_frequency']:.2f} yawns/min",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

        # Always show current blink and yawn count
        cv2.putText(
            frame,
            f"Blink count: {extractor.eye.ear_stats['blinks']}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Yawn count: {extractor.mouth.mar_stats['yawning_count']}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Window Frame Count: {extractor.total_frames} / {extractor.perclos_window}",
            (10, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )

        cv2.imshow("Drowsiness Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
