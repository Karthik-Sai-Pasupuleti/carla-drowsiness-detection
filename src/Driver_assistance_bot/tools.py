"""_summary_"""

import time
from pydantic import BaseModel, Field
from langchain.tools import tool
from .controls import VoiceControl, WheelControlVibration

voice = VoiceControl()
steering = WheelControlVibration()


# schemas
class TextToVoiceInput(BaseModel):
    """Input schema for text-to-voice conversion."""

    text: str = Field(..., description="The text to convert to speech.")


class VibrateSteeringInput(BaseModel):
    """Input schema for steering wheel vibration."""

    intensity: int = Field(30, ge=0, le=60, description="Vibration intensity (0-60).")
    duration: float = Field(0.2, gt=0, description="Duration of vibration in seconds.")


# tools


@tool("voice_alert", args_schema=TextToVoiceInput, return_direct=True)
def text_to_voice(text: str) -> str:
    """Convert text to speech.

    Args:
        text (TextToVoiceInput): The text to convert to speech.

    Returns:
        str: A message indicating the text has been converted to speech.
    """
    try:
        voice.text_to_speech(text)
    except Exception as e:
        return f"Error converting text to speech: {e}"
    return f"Text '{text}' has been converted to speech."


@tool("steering_vibration", args_schema=VibrateSteeringInput, return_direct=True)
def vibrate_steering_wheel(duration: float, intensity: int) -> str:
    """Vibrate the steering wheel.

    Args:
        input (VibrateSteeringInput): The input containing vibration parameters.

    Returns:
        str: A message indicating the steering wheel has been vibrated.
    """
    try:
        steering.vibrate(duration, intensity)
        time.sleep(0.2)
        steering.vibrate(duration=0)
    except Exception as e:
        return f"Error vibrating steering wheel: {e}"
    return f"Steering wheel vibrated with intensity {intensity} for {duration} seconds."
