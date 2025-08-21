"""_summary_

Raises:
    NotImplementedError: _description_
    NotImplementedError: _description_
    NotImplementedError: _description_

Returns:
    _type_: _description_
"""

from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
import time

from .controls import VoiceControl, WheelControlVibration

# Voice Alert Tool


class VoiceAlertInput(BaseModel):
    message: str = Field(..., description="The message to speak out loud.")


class VoiceAlertTool(BaseTool):
    name: str = "voice_alert"
    description: str = "Speak a message to alert the driver via text-to-speech."
    args_schema: Type[BaseModel] = VoiceAlertInput

    def _run(self, message: str) -> str:
        voice = VoiceControl()
        voice.text_to_speech(message)
        return f"Voice alert spoken: {message}"


# Steering Vibration Tool


class SteeringVibrationInput(BaseModel):
    intensity: int = Field(30, ge=0, le=60, description="Vibration intensity (0-60).")
    duration: float = Field(0.2, gt=0, description="Duration of vibration in seconds.")


class SteeringVibrationTool(BaseTool):
    name: str = "steering_vibration"
    description: str = "Trigger steering wheel vibration to wake up the driver."
    args_schema: Type[BaseModel] = SteeringVibrationInput

    def _run(self, intensity: int, duration: float) -> str:
        wheel = WheelControlVibration()
        wheel.vibrate(duration=duration, intensity=intensity)
        time.sleep(duration)
        wheel.vibrate(duration=0)  # stop
        return f"Steering vibration triggered (intensity={intensity}, duration={duration}s)."
