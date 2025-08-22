"""This module contains the driver assistance bot which takes the input such as driver eye
blink frequency, PERCLOS, Yawning frequency and alert the driver if he/she gets drowsy.
"""

from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field, StrictFloat
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


class BotConfig(BaseModel):
    """Configuration for the driver assistance bot."""

    model_id: str
    prompts: Dict[str, Any]
    schema: Dict[str, Any]


class Bot:
    """Contains all schemas for the driver assistance bot."""

    def __init__(self, config: BotConfig):
        model_id = config.model_id
        prompts = config.prompts
        self.schema = config.schema
        self.llm = ChatOllama(model=model_id, temperature=0)

        # Load prompts from TOML
        system_prompt = prompts["bot_prompt"]["SYSTEM"]
        user_prompt = prompts["bot_prompt"]["USER"]

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("user", user_prompt),
            ]
        )

        # self.str_llm = llm.with_structured_output(schema)

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

    class OutputWithActions(BaseModel):
        """Output class with actions for driver assistance bot"""

        drowsiness_level: Literal["low", "medium", "high"] = Field(
            description="Detected drowsiness risk level."
        )
        reasoning: str = Field(
            description="Explanation based on input metrics that led to the decision."
        )
        actions: List[Literal["speaker", "fan", "steering_vibration"]] = Field(
            description="List of actions to trigger in response to detected drowsiness."
        )

    class OutputWithoutActions(BaseModel):
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
