import asyncio
from pathlib import Path
from utils import load_toml, load_json
from tools import vibrate_steering_wheel, text_to_voice
from bot import Bot, BaseBot


# Create bot instance at module load
prompt_file = Path("src") / "Driver_assistance_bot" / "configs" / "prompt.toml"
schema_file = Path("src") / "Driver_assistance_bot" / "configs" / "schema.json"

prompt_cfg = load_toml(prompt_file)
schema_cfg = load_json(schema_file)  # Uncomment if schema mode is needed

MODEL_ID = "llama3.1:8b"  # or "phi3:mini"

config = Bot.BotConfig(
    model_id=MODEL_ID,
    system_prompt=prompt_cfg["bot_prompt"]["SYSTEM"],
    user_prompt=prompt_cfg["bot_prompt"]["USER"],
    schema=schema_cfg,  # or schema_cfg if structured mode
    tools=[text_to_voice, vibrate_steering_wheel],
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
    # Example telemetry
    sample_input = {
        "perclos": 0.6,
        "blink_rate": 12,
        "yawn_freq": 5,
        "sdlp": 0.8,
        "steering_entropy": 0.6,
        "steering_reversal_rate": 10,
    }
    response_ = main(sample_input)
    print(response_)
