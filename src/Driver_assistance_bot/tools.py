import asyncio
import time
import logging
import threading
from pydantic import BaseModel, Field
from langchain.tools import tool
from controls import VoiceControl, WheelControlVibration

voice = VoiceControl()

# configure logging once at the start of your program
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


class VoiceAlertSchema(BaseModel):
    text: str = Field(..., description="The text to convert to speech.")


class VibrateSteeringSchema(BaseModel):
    duration: float = Field(..., gt=0, le=3, description="Duration in seconds")
    intensity: int = Field(..., ge=0, le=60, description="Vibration intensity")


def voice_alert(text: str) -> str:
    """Alert the driver via text-to-voice conversion."""

    def _speak():
        try:
            print("voice command initialize voice alert")
            voice.text_to_speech(text)
            print("voice command initialization complete")
            return f"Voice alert started for text: '{text}'"
        except Exception as e:
            logging.error(f"[Voice Alert Error] {e}")

    threading.Thread(target=_speak, daemon=True).start()


# ---------------- Vibrate Steering ----------------
@tool(args_schema=VibrateSteeringSchema)
def vibrate_steering_wheel(duration: float, intensity: int) -> str:
    """
    Vibrates the steering wheel to alert the driver.
    The duration of the vibration must be 3 seconds or less.
    The intensity must be 60 or less.
    """

    def _vibrate():
        try:
            print("initialize vibrate steering wheel")
            steering = WheelControlVibration()
            print("end=initialize vibrate steering wheel")
            steering.vibrate(duration=duration, intensity=intensity)
            time.sleep(duration)
            steering.vibrate(duration=0)  # stop vibration
            print("end=vibrate steering wheel")
            return f"Steering wheel vibration started with intensity {intensity} for {duration} seconds."
        except Exception as e:
            logging.error(f"[Steering Wheel Error] {e}")

    threading.Thread(target=_vibrate, daemon=True).start()
