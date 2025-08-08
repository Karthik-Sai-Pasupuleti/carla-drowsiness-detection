"""This module contains the driver assistance bot which takes the input such as driver eye
blink frequency, PERCLOS, Yawning frequency and alert the driver if he/she gets drowsy.
"""

from typing import List, Literal, Optional
import toml
from pydantic import BaseModel, Field, StrictFloat
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM


class Input(BaseModel):
    """
    Input data model for the driver drowsiness detection system.

    Args:
        BaseModel (pydantic.BaseModel): Provides data validation and parsing.
    """

    perclos: Optional[StrictFloat] = Field(
        default=None,
        description="Percentage of time eyes are closed, used to measure drowsiness.",
    )
    blink_rate: Optional[StrictFloat] = Field(
        default=None,
        description="Number of eye blinks per minute, detected from video.",
    )
    yawn_freq: Optional[StrictFloat] = Field(
        default=None, description="Number of yawns per minute, detected from video."
    )
    sdlp: Optional[StrictFloat] = Field(
        default=None, description="Standard deviation of lane position in meters."
    )
    steering_entropy: Optional[StrictFloat] = Field(
        default=None, description="Unpredictability measure of steering movements."
    )
    steering_reversal_rate: Optional[StrictFloat] = Field(
        default=None,
        description="Number of significant steering direction changes per minute.",
    )


class Output(BaseModel):
    """
    Output data model for the driver drowsiness detection system.

    Args:
        BaseModel (pydantic.BaseModel): Provides data validation and parsing.
    """

    drowsiness_level: Literal["low", "medium", "high"] = Field(
        description="Detected drowsiness risk level."
    )
    reasoning: str = Field(
        description="Brief explanation based on input metrics that led to the decision."
    )
    actions: List[Literal["speaker", "fan", "steering_vibration"]] = Field(
        description="List of actions to trigger in response to detected drowsiness."
    )


class Bot:
    """Driver drowsiness detection bot"""

    def __init__(self):
        self.llm = OllamaLLM(model="gpt-oss:20b")

        # Load prompts from TOML
        prompts = toml.load("prompts.toml")
        system_prompt = prompts["SYSTEM"]
        user_prompt = prompts["USER"]

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("user", user_prompt),
            ]
        )

    def invoke(self, input_data: Input) -> Output:
        """
        Invoke the LLM
        Args:
            input_data (Input): Driver monitoring and vehicle metrics
        Returns:
            Output: LLM response parsed into Output model
        """

        formatted_messages = self.prompt.format_messages(input_data)

        raw_response = self.llm.invoke(formatted_messages)

        output = Output.model_validate_json(raw_response)

        return output
