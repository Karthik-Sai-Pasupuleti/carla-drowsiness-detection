"""_summary_

Returns:
    _type_: _description_
"""

from pprint import pprint
from pathlib import Path
from bot import Bot, BotConfig
from utils import load_toml, load_json
from tools import vibrate_steering_wheel, text_to_voice


class DriverAssistanceBot(Bot):
    """Driver drowsiness detection bot"""

    # In your __init__ method, define the tool mapping<<<<
    def __init__(self, config: BotConfig):
        super().__init__(config)
        # Define your tools and create a mapping
        self.tools = [vibrate_steering_wheel, text_to_voice]
        self.tool_map = {tool.name: tool for tool in self.tools}
        self.model = self.llm.bind_tools(self.tools)
        self.model = self.model.with_structured_output(self.schema)

    # In your invoke method, use the mapping to get the tool and then invoke it
    async def invoke(self, input_data: Bot.Input) -> Bot.OutputWithoutActions:
        formatted_messages = self.prompt.invoke({"drowsiness_metrics": input_data})
        raw_response = self.model.invoke(formatted_messages)
        output = Bot.OutputWithoutActions.model_validate(raw_response)

        if output.tool_calls:
            for tool_call in output.tool_calls:
                tool_name = tool_call["name"]
                args = tool_call["args"]

                # Check if the tool exists in your map
                if tool_name in self.tool_map:
                    tool_instance = self.tool_map[tool_name]
                    print(f"Executing tool: {tool_name} with args: {args}")
                    # The crucial change: use the `invoke` method of the tool
                    tool_result = tool_instance.invoke(args)
                    print(f"Tool execution result: {tool_result}")
                else:
                    print(f"Warning: Tool '{tool_name}' not found in tool map.")

        return raw_response


if __name__ == "__main__":
    prompt_file = Path("src") / "Driver_assistance_bot" / "configs" / "prompt.toml"
    schema_file = Path("src") / "Driver_assistance_bot" / "configs" / "schema.json"
    prompt_ = load_toml(prompt_file)
    schema_ = load_json(schema_file)
    MODEL_ID = "phi3:mini"  # "llama3.2:latest"/
    # MODEL_ID = "gpt-oss:20b"

    config_ = BotConfig(model_id=MODEL_ID, prompts=prompt_, schema=schema_)
    bot_ = DriverAssistanceBot(config_)

    llm_input = {
        "perclos": 0.6,
        "blink_rate": 12,
        "yawn_freq": 5,
        "sdlp": 0.8,
        "steering_entropy": 0.6,
        "steering_reversal_rate": 10,
    }
    result = bot_.invoke(llm_input)
    pprint(result)
