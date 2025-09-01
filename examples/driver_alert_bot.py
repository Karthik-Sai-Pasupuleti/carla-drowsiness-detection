"""This module has an example implementation of a driver alert bot."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.Driver_assistance_bot.utils import load_toml, load_json
from src.Driver_assistance_bot.tools import vibrate_steering_wheel, voice_alert
from src.Driver_assistance_bot.bot import Bot, BaseBot


# Create bot instance at module load
prompt_file = Path("src") / "Driver_assistance_bot" / "configs" / "prompt.toml"
schema_file = Path("src") / "Driver_assistance_bot" / "configs" / "schema.json"

prompt_cfg = load_toml(prompt_file)
schema_cfg = load_json(schema_file)  # Uncomment if schema mode is needed

MODEL_ID = "llama3.1:8b"

config = Bot.BotConfig(
    model_id=MODEL_ID,
    system_prompt=prompt_cfg["SYSTEM"],
    user_prompt=prompt_cfg["USER"],
    output_schema=schema_cfg,  # or schema_cfg if structured mode
    tools=[voice_alert, vibrate_steering_wheel],
    temperature=0.1,
)

bot_instance = BaseBot(config=config)


def main(telemetry: dict):
    """
    Entry point for invoking the bot with telemetry input and executing tool calls.
    """

    response = bot_instance.invoke(Bot.Input(**telemetry))

    return response


if __name__ == "__main__":
    # Example telemetry input
    sample_input = {
        "perclos": 0.8,
        "blink_rate": 12,
        "yawn_freq": 5,
        "sdlp": 0.8,
        "steering_entropy": 0.6,
        "steering_reversal_rate": 10,
    }
    response_ = main(sample_input)
    print("drowsiness_level:", response_.drowsiness_level)
    print("reasoning:", response_.reasoning)
    print("tool_calls:", response_.tool_calls)
