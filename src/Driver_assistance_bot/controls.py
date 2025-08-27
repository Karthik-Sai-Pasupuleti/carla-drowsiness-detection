"""This module provides the Logitech G29 steering wheel vibration control and
text-to-speech functionality using pyttsx3."""

import os
import time
import threading
from threading import Lock
import pyttsx3
import logging
import asyncio


class VoiceControl:
    """Handles text-to-speech conversion safely by creating a fresh engine each time."""

    def __init__(self):
        self.rate = 145
        self.volume = 1.0
        self.lock = threading.Lock()

    def text_to_speech(self, text: str):
        """Speak text asynchronously in a thread-safe way."""

        def speak():
            try:
                with self.lock:

                    engine = pyttsx3.init()
                    engine.setProperty("rate", self.rate)
                    engine.setProperty("volume", self.volume)
                    engine.say(text)
                    engine.runAndWait()
                    engine.stop()  # ensure cleanup
            except Exception as e:
                logging.error(f"[VoiceControl Error] {e}")

        threading.Thread(target=speak, daemon=True).start()


class WheelControlVibration:
    """
    This class control the Logitech G29 steering wheel vibration.
    It waits for a key event and triggers vibration as long as key is pressed.

    Requirements:
    - evdev: For interacting with input devices.
    - os: For interacting with the file system (raw HID access).
    - time: For controlling vibration duration.
    - threading.Lock: To ensure thread-safe access to the raw device.
    - pygame:  For event handling and joystick input (if needed for key presses).
    """

    HIDRAW_DEVICE = "logitech_raw"  #

    def __init__(self):
        """
        Initializes the WheelControlVibration object. Opens the raw
        HID device for the Logitech G29.
        """
        self.raw_dev = None
        try:
            self.raw_dev = os.open(
                "/dev/{}".format(WheelControlVibration.HIDRAW_DEVICE), os.O_RDWR
            )
            print("Logitech G29 initialized successfully.")
        except OSError as e:
            print(
                f"Failed to initialize Logitech G29 device {self.HIDRAW_DEVICE}. Vibration will not work.  Error: {e}"
            )
            self.raw_dev = None  # Important: Set to None to prevent errors later.

        self._steering_wheel_write_lock = Lock()
        self.vibration_active = False  # Track if vibration is active

    def vibrate(self, duration=0.05, intensity=25):
        """
        Triggers the vibration of the Logitech G29 steering wheel.

        Args:
            duration (float): The duration of the vibration in seconds. Increase value for continuous.
            intensity (int): The intensity of the vibration (0-127). I have set the Default to 50.
        """
        if self.raw_dev is None:
            print("Raw device not initialized.  Cannot vibrate.")
            return

        # Clamp intensity to the valid range.
        intensity = max(0, min(intensity, 60))

        # Vibrate uses slot F1.
        force_altitude = intensity  # Use intensity directly.

        with self._steering_wheel_write_lock:
            try:
                if duration > 0 and not self.vibration_active:
                    # Start vibration.
                    os.write(
                        self.raw_dev,
                        bytearray(
                            [
                                0x21,
                                0x06,
                                128 + force_altitude,
                                128 - force_altitude,
                                8,
                                8,
                                0x0F,
                            ]
                        ),
                    )
                    self.vibration_active = True
                elif duration == 0 and self.vibration_active:
                    os.write(
                        self.raw_dev,
                        bytearray([0x23, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
                    )  # Stop vibration
                    self.vibration_active = False
            except OSError as e:
                print(f"Error writing to raw device to vibrate: {e}")

    def close(self):
        """
        Closes the connection to the Logitech Steering Wheel device.
        """
        if self.raw_dev:
            try:
                os.close(self.raw_dev)
                self.raw_dev = None
                print("Raw device closed.")
            except OSError as e:
                print(f"Error closing raw device: {e}")

    def __del__(self):
        """
        Destructor to ensure devices is closed
        """
        self.close()


if __name__ == "__main__":
    # Example usage
    # Wheel Control Vibration
    wheel_vibration = WheelControlVibration()
    wheel_vibration.vibrate()
    time.sleep(0.1)
    wheel_vibration.vibrate(duration=0)  # Stop vibration
    wheel_vibration.close()
    # Text to Speech Control
    bot = VoiceControl()
    bot.text_to_speech("Hello, this is a test of the text to speech functionality.")
