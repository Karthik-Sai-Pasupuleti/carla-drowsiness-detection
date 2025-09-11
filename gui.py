"""_summary_"""

import sys
import time
import threading
import random
from collections import deque
import cv2
import msvc_runtime
from PyQt5.QtWidgets import (
    QMessageBox,
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QFrame,
    QSizePolicy,
    QSplitter,
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
import numpy as np

# Feature extraction utilities
from src.feature_extraction.utils import (
    calculate_perclos,
    calculate_blink_frequency,
    calculate_yawn_frequency,
    vehicle_feature_extraction,
)
from src.feature_extraction.camera_features import CameraBasedFeatureExtractor

# from src.carla_api.manual_control_steering_wheel import carla_steering_wheel


shared_data = {
    "camera_frames": deque(maxlen=10000),
    "vehicle_data": deque(maxlen=10000),
}  # prevent unbounded growth


class CameraThread(QThread):
    # Use object for numpy frames to avoid PyQt type issues
    frame_update = pyqtSignal(object)
    metrics_update = pyqtSignal(float, float)  # timestamp, EAR, MAR

    def __init__(self, data):
        super().__init__()
        self.data = data
        self.running = True
        self.camera_extractor = CameraBasedFeatureExtractor()
        self.fps = None

    def run(self):
        try:
            extractor = CameraBasedFeatureExtractor()
            cap = cv2.VideoCapture(0)
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            if self.fps is None:
                self.fps = 30

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                current_time = time.time()
                ear, mar = extractor.process_frame(frame)
                self.frame_update.emit(frame)
                if ear is not None and mar is not None:
                    self.data["camera_frames"].append((current_time, ear, mar))
                    self.metrics_update.emit(ear, mar)

        except Exception as e:  # pylint: disable=broad-exception-caught
            print("Error in Camera Feature Module:", e)

        finally:
            cap.release()
            cv2.destroyAllWindows()

    def stop(self):
        self.running = False


class CarlaThread(QThread):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.running = True

    def run(self):
        # carla_steering_wheel(self.data)
        pass

    def stop(self):
        self.running = False


class SinglePlotCanvas(FigureCanvas):
    def __init__(self, title, y_label, parent=None, width=4, height=2.5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

        self.ax.set_title(title)
        self.ax.set_xlabel("Time (frames)")
        self.ax.set_ylabel(y_label)
        self.ax.grid(True)

        (self.line,) = self.ax.plot([], [], label=title)
        self.ax.legend()

        self.time_data = []
        self.value_data = []
        # Add a continuous counter for the x-axis
        self.frame_counter = 0

    def update_plot(self, value):
        # Use the continuous counter for the x-axis
        self.time_data.append(self.frame_counter)
        self.value_data.append(value)

        # Increment the counter
        self.frame_counter += 1

        # keep only last 100 points
        self.time_data = self.time_data[-100:]
        self.value_data = self.value_data[-100:]

        # Adjust the x-axis limits to prevent it from looking stuck
        self.ax.set_xlim(
            self.time_data[0],
            self.time_data[-1] if len(self.time_data) > 1 else self.time_data[0] + 1,
        )

        self.line.set_data(self.time_data, self.value_data)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw_idle()


# ---------------- Main GUI ----------------
class DrowsinessGUI(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Driver Drowsiness Detection")
        self.setGeometry(50, 50, 1200, 700)
        self.init_ui()
        self.data_lock = threading.Lock()

        self.shared_data = {
            "camera_frames": deque(maxlen=1000),
            "vehicle_data": deque(maxlen=1000),
        }

        # Camera Thread
        self.cam_thread = CameraThread(self.shared_data)
        self.cam_thread.frame_update.connect(self.update_camera)
        self.cam_thread.metrics_update.connect(self.update_ear_mar)
        self.cam_thread.start()

        # Carla Thread
        self.carla_thread = CarlaThread(self.shared_data)
        self.carla_thread.start()

        # Periodic UI updater
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_ui)
        self.timer.start(200)

        # 60s reminder timer (instead of auto-save)
        self.reminder_timer = QTimer()
        self.reminder_timer.timeout.connect(self.show_reminder)
        self.reminder_timer.start(60000)  # remind every 60s

        self.results_df = pd.DataFrame(
            columns=["timestamp", "PERCLOS", "Blink Rate", "Yawn Rate", "SDLP", "SRR"]
        )
        self.action_events = list()
        self.drowsiness_label = None

        # Store the last processed row until user submits
        self.pending_row = None

    def show_reminder(self):
        if self.pending_row is not None:
            # Already waiting for submission, don’t annoy user again
            return

        reply = QMessageBox.information(
            self,
            "Reminder",
            "Please add drowsiness labels and submit within this 60s window.",
            QMessageBox.Ok,
        )
        # Process the last 60s data but don’t save yet
        self.prepare_time_window()

        # Start the 60-second countdown
        self.countdown_time = 60
        self.timer_label.setText(f"Time left: {self.countdown_time}s")
        self.countdown_timer.start(1000)  # update every 1 second

    def refresh_ui(self):
        with self.data_lock:
            camera_frames = list(self.shared_data["camera_frames"])
            vehicle_data = list(self.shared_data["vehicle_data"])

        if camera_frames:
            _, ear, mar = camera_frames[-1]
            self.update_ear_mar(ear, mar)
        else:
            ear, mar = 0, 0

        if vehicle_data:
            _, steering_angle, _ = vehicle_data[-1]
        else:
            steering_angle = 0

        # update separate plots
        self.ear_plot.update_plot(ear)
        self.mar_plot.update_plot(mar)
        self.steering_plot.update_plot(steering_angle)

    def prepare_time_window(self):
        """Collect last 60s window but wait for user to submit."""
        with self.data_lock:
            camera_snapshot = list(self.shared_data["camera_frames"])
            vehicle_snapshot = list(self.shared_data["vehicle_data"])
            drowsiness_label = self.drowsiness_label
            actions = self.action_events
            self.shared_data["camera_frames"].clear()
            self.shared_data["vehicle_data"].clear()
            self.drowsiness_label = None
            self.action_events = []

        # Process camera snapshot
        if camera_snapshot:
            camera_time, ear_list, mar_list = zip(*camera_snapshot)
            camera_time = np.array(camera_time)
            ear_list = np.array(ear_list)
            mar_list = np.array(mar_list)

            perclos = calculate_perclos(ear_list, 0.2, 3)
            blink_rate = calculate_blink_frequency(ear_list, 0.2, self.cam_thread.fps)
            yawn_rate = calculate_yawn_frequency(
                mar_list, 0.4, int(self.cam_thread.fps * 4), self.cam_thread.fps
            )
        else:
            camera_time, ear_list, mar_list = [], [], []
            perclos, blink_rate, yawn_rate = 0, 0, 0

        if vehicle_snapshot:
            vehicle_time, steering_angle, lane_position = zip(*vehicle_snapshot)
            vehicle_time = np.array(vehicle_time)
            steering_angle = np.array(steering_angle)
            lane_position = np.array(lane_position)
            entropy, steering_rate, sdlp = vehicle_feature_extraction(
                steering_angle, lane_position, 60
            )
            vehicle_metrics = {"SDLP": sdlp, "SRR": steering_rate, "Entropy": entropy}
        else:
            vehicle_time, steering_angle, lane_position = [], [], []
            vehicle_metrics = {}

        # Prepare but don't save yet
        self.pending_row = {
            "timestamp": pd.Timestamp.now(),
            "camera_timestamps": camera_time,
            "ear_values": ear_list,
            "mar_values": mar_list,
            "vehicle_timestamps": vehicle_time,
            "steering_angle": steering_angle,
            "lane_position": lane_position,
            "PERCLOS": perclos,
            "Eye Blink Rate": blink_rate,
            "Yawning Rate": yawn_rate,
            "SDLP": vehicle_metrics.get("SDLP"),
            "SRR": vehicle_metrics.get("SRR"),
            "Entropy": vehicle_metrics.get("Entropy"),
            "Actions": actions,
            "Drowsiness Level": drowsiness_label,
        }

    def _on_action_clicked(self, action_name: str):
        if action_name not in self.action_events:  # keep the list unique
            self.action_events.append(action_name)
        print("Selected actions:", self.action_events)

    def _on_drowsiness_label(self, level: str):
        self.drowsiness_label = level
        print("Selected drowsiness level:", self.drowsiness_label)

    def update_countdown(self):
        self.countdown_time -= 1
        if self.countdown_time <= 0:
            self.countdown_timer.stop()
            self.timer_label.setText("Time left: 0s")
            QMessageBox.information(
                self, "Time's Up", "60-second labeling window ended."
            )
            # Optional: auto-submit if not already submitted
            # self.submit_data()
        else:
            self.timer_label.setText(f"Time left: {self.countdown_time}s")

    def submit_data(self):
        if self.pending_row is None:
            QMessageBox.warning(
                self, "No Data", "No 60s window data ready for submission."
            )
            return

        self.results_df = pd.concat(
            [self.results_df, pd.DataFrame([self.pending_row])],
            ignore_index=True,
        )
        self.update_metrics(self.pending_row)
        print("60-second window submitted and saved.")
        self.pending_row = None  # reset

        # Stop countdown if running
        if self.countdown_timer.isActive():
            self.countdown_timer.stop()
            self.timer_label.setText("Time left: 60s")

    def init_ui(self):
        main_layout = QHBoxLayout()  # Horizontal split: Left | Center | Right

        # ----------------- LEFT PANEL -----------------
        left_layout = QVBoxLayout()

        # --- Facial & Vehicle Features ---
        metrics_layout = QGridLayout()
        metrics_group = QGroupBox("Features")
        metrics_group.setLayout(metrics_layout)

        facial_features = ["EAR", "MAR", "Blinks", "Yawns", "PERCLOS"]
        vehicle_features = ["SDLP", "SRR", "Entropy"]
        self.metric_labels = {}
        # --- 60s Countdown Timer ---
        self.timer_label = QLabel("Time left: 60s")
        self.timer_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: red;"
        )
        left_layout.addWidget(self.timer_label, alignment=Qt.AlignCenter)
        self.countdown_time = 60  # 60 seconds
        self.countdown_timer = QTimer()
        self.countdown_timer.timeout.connect(self.update_countdown)

        # Facial Features Column
        for i, feat in enumerate(facial_features):
            lbl_name = QLabel(feat)
            lbl_value = QLabel("0.0")
            lbl_value.setStyleSheet("font-weight: bold")
            metrics_layout.addWidget(lbl_name, i, 0)
            metrics_layout.addWidget(lbl_value, i, 1)
            self.metric_labels[feat] = lbl_value

        # Vehicle Features Column
        for i, feat in enumerate(vehicle_features):
            lbl_name = QLabel(feat)
            lbl_value = QLabel("0.0")
            lbl_value.setStyleSheet("font-weight: bold")
            metrics_layout.addWidget(lbl_name, i, 2)
            metrics_layout.addWidget(lbl_value, i, 3)
            self.metric_labels[feat] = lbl_value

        left_layout.addWidget(metrics_group)

        # --- Drowsiness Level ---
        drowsiness_group = QGroupBox("Drowsiness Level")
        drowsiness_layout = QHBoxLayout()
        for lvl in ["High", "Moderate", "Low"]:
            btn = QPushButton(lvl)
            btn.clicked.connect(lambda _, l=lvl: self._on_drowsiness_label(l))
            btn.setFixedHeight(50)
            drowsiness_layout.addWidget(btn)
        drowsiness_group.setLayout(drowsiness_layout)
        left_layout.addWidget(drowsiness_group)

        # --- Actions ---
        action_group = QGroupBox("Actions")
        action_layout = QHBoxLayout()
        for act in ["Fan", "Voice", "Steering Vibration"]:
            btn = QPushButton(act)
            btn.clicked.connect(lambda _, a=act: self._on_action_clicked(a))
            btn.setFixedHeight(50)
            action_layout.addWidget(btn)
        action_group.setLayout(action_layout)
        left_layout.addWidget(action_group)

        # --- Submit Button ---
        submit_btn = QPushButton("Submit")
        submit_btn.clicked.connect(self.submit_data)
        submit_btn.setFixedHeight(50)
        left_layout.addWidget(submit_btn, alignment=Qt.AlignCenter)

        # ----------------- CENTER PANEL -----------------
        center_layout = QVBoxLayout()
        self.cam_label = QLabel()
        self.cam_label.setMinimumSize(800, 600)
        self.cam_label.setFrameShape(QFrame.Box)
        self.cam_label.setAlignment(Qt.AlignCenter)
        self.cam_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        center_layout.addWidget(self.cam_label, alignment=Qt.AlignCenter)

        # ----------------- RIGHT PANEL -----------------
        right_layout = QVBoxLayout()
        self.steering_plot = SinglePlotCanvas("Steering Angle", "Degrees", self)
        self.mar_plot = SinglePlotCanvas("MAR", "Value", self)
        self.ear_plot = SinglePlotCanvas("EAR", "Value", self)

        for plot in [self.steering_plot, self.mar_plot, self.ear_plot]:
            right_layout.addWidget(plot)

        # ----------------- COMBINE PANELS -----------------
        main_layout.addLayout(left_layout, stretch=2)
        main_layout.addLayout(center_layout, stretch=3)
        main_layout.addLayout(right_layout, stretch=3)

        self.setLayout(main_layout)

    def update_camera(self, frame):
        # Convert BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w

        # Create QImage and COPY to decouple from numpy memory
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)

        # Scale to label size for consistent display
        pix = pix.scaled(
            self.cam_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.cam_label.setPixmap(pix)

    def update_metrics(self, metrics):
        # Only update keys that exist in labels
        for k, v in metrics.items():
            if k in self.metric_labels:
                self.metric_labels[k].setText(f"{v}")

    def update_ear_mar(self, ear, mar):
        if "EAR" in self.metric_labels:
            self.metric_labels["EAR"].setText(f"{ear:.2f}")
        if "MAR" in self.metric_labels:
            self.metric_labels["MAR"].setText(f"{mar:.2f}")

    def closeEvent(self, event):
        # Stop threads cleanly
        if hasattr(self, "cam_thread"):
            self.cam_thread.stop()
            self.cam_thread.wait(2000)
        if hasattr(self, "metrics_thread"):
            self.metrics_thread.stop()
            self.metrics_thread.wait(2000)
        event.accept()


# ---------------- Run Application ----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = DrowsinessGUI()
    gui.show()
    sys.exit(app.exec_())
