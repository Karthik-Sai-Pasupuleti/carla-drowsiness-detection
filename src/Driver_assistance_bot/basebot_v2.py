"""_summary_

Returns:
    _type_: _description_
"""

import asyncio
from pprint import pprint
from pathlib import Path
from Bot import Bot, BotConfig
from utils import load_toml, load_json
from tools import vibrate_steering_wheel, text_to_voice


class DriverAssistanceBot(Bot):
    """Driver drowsiness detection bot"""

    def __init__(self, config: BotConfig):
        super().__init__(config)

        # Register tools
        self.tools = [vibrate_steering_wheel, text_to_voice]
        self.tool_map = {tool.name: tool for tool in self.tools}
        print("Tool map:", self.tool_map)

        # Bind tools so the model can actually call them
        self.model = self.llm.bind_tools(self.tools)

        # self.model = self.model.with_structured_output(self.schema)

    async def invoke(self, input_data: Bot.Input) -> str:
        # Format messages from your prompt template
        formatted_messages = await self.prompt.ainvoke(
            {"drowsiness_metrics": input_data}
        )

        # Ask LLM to process input (it may return tool calls)
        raw_response = await self.model.ainvoke(formatted_messages)

        return raw_response


if __name__ == "__main__":
    prompt_file = Path("src") / "Driver_assistance_bot" / "configs" / "prompt.toml"
    schema_file = Path("src") / "Driver_assistance_bot" / "configs" / "schema.json"
    prompt_ = load_toml(prompt_file)
    schema_ = load_json(schema_file)
    MODEL_ID = "llama3.1:8b"  # "phi3:mini"

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

    async def main():
        result = await bot_.invoke(llm_input)
        pprint(result.dict())

    asyncio.run(main())
