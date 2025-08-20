"""This module contains the functions to control the electronics such as fan,
steering wheel (vibration), speaker (announcement)
"""

import pyttsx3

import evdev
from evdev import ecodes, InputDevice


def steering_force_feedback(device):

    evtdev = InputDevice(device)
    val = 65535  # val \in [0,65535]
    evtdev.write(ecodes.EV_FF, ecodes.FF_AUTOCENTER, val)


class bot_controls:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 145)
        self.engine.setProperty("volume", 1.0)  # volume level between 0 to 1.

    def text_to_speech(self, text: str):
        """convert the text to speech using pyttsx library.

        Args:
            text (str): text from the bot spoken aloud.
        """
        self.engine.say(text)
        self.engine.runAndWait()
        self.engine.stop()
