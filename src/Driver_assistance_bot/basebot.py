"""_summary_

Returns:
    _type_: _description_
"""

from pprint import pprint
from pathlib import Path
from .Bot import Bot, BotConfig
from .utils import load_toml, load_json
from .tools import VoiceAlertTool, SteeringVibrationTool


class DriverAssistanceBot(Bot):
    """Driver drowsiness detection bot"""

    def __init__(self, config: BotConfig):
        """Initialize the driver assistance agent.

        Args:
            config (BotConfig): Configuration for the bot.
        """
        super().__init__(config)
        tools = [VoiceAlertTool(), SteeringVibrationTool()]
        self.model = self.llm.bind_tools(tools)
        self.model_with_tools = self.model.with_structured_output(self.schema)

    def invoke(self, input_data: Bot.Input) -> Bot.OutputWithoutActions:
        """
        Invoke the LLM
        Args:
            input_data (Input): Driver monitoring and vehicle metrics
        Returns:
            Output: LLM response parsed into Output model
        """
        formatted_messages = self.prompt.invoke({"drowsiness_metrics": input_data})

        raw_response = self.model_with_tools.invoke(formatted_messages)
        output = Bot.OutputWithoutActions.model_validate(raw_response)
        return output


if __name__ == "__main__":
    prompt_file = Path("src") / "Driver_assistance_bot" / "configs" / "prompt.toml"
    schema_file = Path("src") / "Driver_assistance_bot" / "configs" / "schema.json"
    prompt_ = load_toml(prompt_file)
    schema_ = load_json(schema_file)
    MODEL_ID = "llama3.1:8b"  # "llama3.2:latest"

    config_ = BotConfig(model_id=MODEL_ID, prompts=prompt_, schema=schema_)
    bot_ = DriverAssistanceBot(config_)

    llm_input = {
        "perclos": 30,
        "blink_rate": 60,
        "yawn_freq": 20,
        "sdlp": 0.1,
        "steering_entropy": 0.6,
        "steering_reversal_rate": 10,
    }
    result = bot_.invoke(llm_input)
    pprint(result.dict())
