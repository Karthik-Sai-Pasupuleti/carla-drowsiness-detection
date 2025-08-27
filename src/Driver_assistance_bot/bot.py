"""This module contains the driver assistance bot which takes the input such as driver eye
blink frequency, PERCLOS, Yawning frequency and alert the driver if he/she gets drowsy.
"""

from pprint import pprint
import uuid
from typing import (
    List,
    Literal,
    Optional,
    Dict,
    Sequence,
    Callable,
    Any,
)
from pydantic import BaseModel, Field, StrictFloat, ValidationError
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode
from langchain.schema import AIMessage
from langgraph.prebuilt import create_react_agent


class Bot:
    """Bot schemas"""

    class BotConfig(BaseModel):
        """Configuration for the driver assistance bot."""

        model_id: str
        system_prompt: str | None = None
        user_prompt: str | None = None
        schema: Dict[str, Any] | None = Field(
            default=None,
            description="Pydantic schema class for structured output (e.g., OutputWithActions)",
        )
        tools: Sequence[Callable[..., Any]] | None = None
        temperature: float | None = None

    class Input(BaseModel):
        """Input data for the driver assistance bot"""

        perclos: Optional[StrictFloat] = Field(
            default=None, description="Percentage of time eyes are closed."
        )
        blink_rate: Optional[StrictFloat] = Field(
            default=None, description="Number of eye blinks per minute."
        )
        yawn_freq: Optional[StrictFloat] = Field(
            default=None, description="Number of yawns per minute."
        )
        sdlp: Optional[StrictFloat] = Field(
            default=None, description="Standard deviation of lane position (m)."
        )
        steering_entropy: Optional[StrictFloat] = Field(
            default=None, description="Unpredictability measure of steering movements."
        )
        steering_reversal_rate: Optional[StrictFloat] = Field(
            default=None, description="Steering direction changes per minute."
        )

    class Output(BaseModel):
        """Output class without actions for driver assistance bot"""

        drowsiness_level: Literal["low", "medium", "high", "critical"] = Field(
            description="Detected drowsiness risk level."
        )
        reasoning: str = Field(
            description="Explanation based on input metrics that led to the decision."
        )
        tool_calls: List[Dict[str, Any]] = Field(
            description="List of tool calls to execute in response to detected drowsiness."
        )


# class BaseBot(Bot):
#     def __init__(
#         self,
#         config: Bot.BotConfig,
#     ):

#         self.llm = ChatOllama(
#             model=config.model_id,
#             temperature=config.temperature,
#         )

#         # prompts
#         self.prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("system", config.system_prompt),
#                 ("user", config.user_prompt),
#             ]
#         )

#         self.llm.bind_tools(config.tools)
#         self.llm.with_structured_output(config.schema)
#         self.tool_node = ToolNode(config.tools, handle_tool_errors=error_handling)

#     def invoke(self, input_data: Bot.Input):
#         inputs = {"drowsiness_metrics": input_data.dict()}
#         chain = self.prompt | self.llm
#         raw_output = chain.invoke(inputs)

#         try:
#             validated_output = Bot.Output.model_validate_json(raw_output.content)

#             tool_calls = []
#             for call in validated_output.tool_calls:
#                 tool_calls.append(
#                     {
#                         "name": call["name"],
#                         "args": call["args"],
#                         "id": str(uuid.uuid4()),  # unique ID for tracking
#                         "type": "tool_call",
#                     }
#                 )

#             result = self.tool_node.invoke(tool_calls)
#             print(result)
#             return validated_output

#         except ValidationError as e:
#             print("Validation failed:", e)
#             raise e


class BaseBot(Bot):
    def __init__(
        self,
        config: Bot.BotConfig,
    ):

        llm = ChatOllama(
            model=config.model_id,
            temperature=config.temperature,
        )

        # prompts
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", config.system_prompt),
                # ("user", config.user_prompt),
            ]
        )

        self.agent = create_react_agent(
            model=llm,
            tools=config.tools,
            prompt=config.system_prompt,
            name="DriverAssistanceAgent",
        )

    def invoke(self, input_data: Bot.Input):

        raw_ouptut = self.agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f" Drowsiness metrics: {input_data} Based on these metrics, assess the driver’s drowsiness level, reasoning, and recommend alert tools. ",
                    }
                ]
            }
        )

        return raw_ouptut
